-- SPDX-License-Identifier: MPL-2.0
-- Trinite test adapter for the separately pinned NEXUS functional model.
-- No donor source is modified. This is not a Rust refinement proof.
import Nexus.Runtime
open Nexus

def flattenState (s : Lattice.State n) : List Nat :=
  (List.finRange n).flatMap fun c => (List.finRange 12).map fun v => (s c v).val

def emit (tag : String) (seed rounds cells : Nat) (s : Lattice.State cells) : IO Unit :=
  IO.println (String.intercalate "," ([tag] ++
    ([seed, rounds, cells] ++ flattenState s).map toString))

def process (line : String) : IO Unit := do
  let fields := (line.trim.splitOn " ").filter (· != "")
  let some values := fields.mapM String.toNat?
    | throw (IO.userError "non-natural input field")
  match values with
  | seed :: rounds :: cells :: registers =>
    if seed ≥ 2^32 || rounds > 128 || !(cells == 2 || cells == 6) ||
        registers.length != cells * 12 || registers.any (· ≥ 4) then
      throw (IO.userError "input outside frozen conformance bounds")
    if hn : 2 ≤ cells then
      let a := registers.toArray
      let state : Lattice.State cells := fun c v =>
        ⟨a[c.val * 12 + v.val]! % 4, Nat.mod_lt _ (by decide)⟩
      let mut forward := Runtime.snapshot state
      for r in List.range rounds do
        forward := Runtime.snapshot (Lattice.forward cells hn r
          (Runtime.plan cells (UInt32.ofNat seed) r) (Runtime.view forward))
      emit "forward" seed rounds cells (Runtime.view forward)
      let mut restored := forward
      for r in (List.range rounds).reverse do
        restored := Runtime.snapshot (Lattice.inverse cells hn r
          (Runtime.plan cells (UInt32.ofNat seed) r) (Runtime.view restored))
      emit "restored" seed rounds cells (Runtime.view restored)
      let mut inverse := Runtime.snapshot state
      for r in (List.range rounds).reverse do
        inverse := Runtime.snapshot (Lattice.inverse cells hn r
          (Runtime.plan cells (UInt32.ofNat seed) r) (Runtime.view inverse))
      emit "inverse" seed rounds cells (Runtime.view inverse)
      let mut recovered := inverse
      for r in List.range rounds do
        recovered := Runtime.snapshot (Lattice.forward cells hn r
          (Runtime.plan cells (UInt32.ofNat seed) r) (Runtime.view recovered))
      emit "recovered" seed rounds cells (Runtime.view recovered)
    else
      throw (IO.userError "invalid cell count")
  | _ => throw (IO.userError "missing input fields")

def main : IO Unit := do
  let stdin ← IO.getStdin
  let mut count := 0
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    count := count + 1
    if count > 4096 || line.length > 4096 then
      throw (IO.userError "input size bound exceeded")
    process line
