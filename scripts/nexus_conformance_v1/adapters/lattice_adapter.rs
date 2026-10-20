// SPDX-License-Identifier: MPL-2.0
// Test-only access adapter, appended to exact verified donor lattice.rs.
// The public donor API does not accept arbitrary register vectors.
pub fn trinite_states(seed: u32, rounds: u32, input: Vec<u8>) -> Vec<Vec<u8>> {
    assert!(rounds <= 128 && (input.len() == 24 || input.len() == 72));
    assert!(input.iter().all(|v| *v < 4));
    let mut lattice = Lattice::seeded((input.len()/12) as u32, seed).unwrap();
    lattice.registers = input.clone();
    lattice.forward(rounds);
    let forward = lattice.registers.clone();
    lattice.inverse(rounds);
    let restored = lattice.registers.clone();
    lattice.registers = input;
    lattice.inverse(rounds);
    let inverse = lattice.registers.clone();
    lattice.forward(rounds);
    vec![forward, restored, inverse, lattice.registers.clone()]
}
