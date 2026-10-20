// SPDX-License-Identifier: MPL-2.0
mod error;
mod ve;
use std::io::{self, BufRead};
fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let xs: Vec<u32> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
        assert!(xs.len() >= 3);
        let (seed, rounds, cells) = (xs[0], xs[1], xs[2]);
        assert!((cells == 2 || cells == 6) && xs.len() == 3 + cells as usize*12);
        assert!(xs[3..].iter().all(|x| *x < 4));
        let states = ve::lattice::trinite_states(seed, rounds, xs[3..].iter().map(|x| *x as u8).collect());
        for (tag, state) in ["forward", "restored", "inverse", "recovered"].iter().zip(states) {
            let suffix = state.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
            println!("{tag},{seed},{rounds},{cells},{suffix}");
        }
    }
}
