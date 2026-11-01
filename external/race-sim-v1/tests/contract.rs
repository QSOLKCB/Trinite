use std::collections::BTreeSet;
use trinite_race_sim_v1::{
    json, observation, privileged_json, random_word, simulate, Request, Runner,
};
fn request() -> Request {
    Request {
        seed: 7,
        steps: 32,
        track: 1,
        runners: (1..=4)
            .map(|id| Runner {
                id,
                observed: [true; 5],
            })
            .collect(),
        scratches: BTreeSet::new(),
    }
}
#[test]
fn deterministic_replay_and_input_order() {
    let r = request();
    let mut reordered = r.clone();
    reordered.runners.reverse();
    assert_eq!(simulate(&r).unwrap(), simulate(&reordered).unwrap());
    assert_eq!(
        observation(&r, true).unwrap(),
        observation(&reordered, true).unwrap()
    );
    assert_eq!(privileged_json(&r).unwrap(), privileged_json(&r).unwrap());
}
#[test]
fn missing_evidence_is_not_a_scratching() {
    let r = request();
    let mut missing = r.clone();
    missing.runners[0].observed = [false; 5];
    assert_eq!(simulate(&r).unwrap(), simulate(&missing).unwrap());
    let view = observation(&missing, true).unwrap();
    assert!(view.contains("\"id\":1,\"active\":true,\"values\":[0,0,0,0,0],\"observed\":[false,false,false,false,false]"));
    let mut scratched = r.clone();
    scratched.scratches.insert(1);
    assert_eq!(simulate(&scratched).unwrap().len(), 3);
    assert!(simulate(&scratched).unwrap().iter().all(|r| r.id != 1));
    assert!(observation(&scratched, true)
        .unwrap()
        .contains("\"id\":1,\"active\":false"));
    // The earlier forecast view must not disclose a future withdrawal.
    assert_eq!(
        observation(&r, false).unwrap(),
        observation(&scratched, false).unwrap()
    );
}
#[test]
fn scratch_recomputes_field_but_preserves_latents() {
    let r = request();
    let full = simulate(&r).unwrap();
    let mut changed = r.clone();
    changed.scratches.insert(1);
    let reduced = simulate(&changed).unwrap();
    for runner in &reduced {
        let old = full.iter().find(|old| old.id == runner.id).unwrap();
        assert_eq!(runner.latent, old.latent);
    }
    assert!(reduced
        .iter()
        .any(|r| r.distance != full.iter().find(|old| old.id == r.id).unwrap().distance));
}
#[test]
fn views_are_schema_separated_and_valid_json_subset() {
    let r = request();
    for after in [false, true] {
        let view = observation(&r, after).unwrap();
        assert!(json::parse(view.as_bytes()).is_ok());
        for key in [
            "seed",
            "latent",
            "ranking",
            "distance",
            "fatigue",
            "scratches",
        ] {
            assert!(!view.contains(&format!("\"{key}\":")));
        }
    }
    let ground = privileged_json(&r).unwrap();
    assert!(ground.contains("\"privileged\":true"));
    assert!(json::parse(ground.as_bytes()).is_ok());
}
#[test]
fn zero_signal_with_true_mask_exists() {
    let found = (0..100).any(|seed| {
        let mut r = request();
        r.seed = seed;
        observation(&r, true).unwrap().contains(",0,")
    });
    assert!(found);
}
#[test]
fn bounds_duplicate_unknown_and_inactive_fields_rejected() {
    let base = include_str!("../example-request.json");
    assert!(Request::parse(base.as_bytes()).is_ok());
    for bad in [
        base.replace("\"seed\": 7", "\"seed\": true"),
        base.replace("\"seed\": 7", "\"seed\": -1"),
        base.replace("\"seed\": 7", "\"seed\": 4294967296"),
        base.replace("\"steps\": 32", "\"steps\": 0"),
        base.replace("\"steps\": 32", "\"steps\": 257"),
        base.replace("\"track\": 1", "\"track\": 3"),
        base.replace("\"scratches\": []", "\"scratches\": [99]"),
        base.replace("\"scratches\": []", "\"scratches\": [1,1]"),
        base.replace("\"scratches\": []", "\"scratches\": [1,2]"),
        base.replace("\"id\": 2", "\"id\": 1"),
        base.replace("\"id\": 2", "\"id\": 0"),
        base.replace("\"id\": 2", "\"id\": 65536"),
        base.replace("\"track\": 1", "\"track\": 1, \"extra\": 0"),
        base.replace("true, true, true, true, true", "true, true"),
        base.replace("\"id\": 2", "\"id\": 2, \"latent\": 4"),
        base.replace("trinite.race-request.v1", "trinite.race-request.v2"),
    ] {
        assert!(Request::parse(bad.as_bytes()).is_err(), "{bad}");
    }
}
#[test]
fn library_calls_validate_and_budget_is_finite() {
    let mut r = request();
    r.runners = (1..=32)
        .map(|id| Runner {
            id,
            observed: [true; 5],
        })
        .collect();
    r.steps = 256;
    r.seed = u32::MAX;
    r.track = 2;
    let result = simulate(&r).unwrap();
    assert_eq!(result.len(), 32);
    assert!(result.iter().all(|x| (256..=25_600).contains(&x.distance)));
    r.runners.push(Runner {
        id: 33,
        observed: [true; 5],
    });
    assert!(simulate(&r).is_err());
    r = request();
    r.steps = 0;
    assert!(observation(&r, true).is_err());
    r = request();
    r.runners[1].id = r.runners[0].id;
    assert!(simulate(&r).is_err());
}
#[test]
fn ordering_is_distance_then_id() {
    for seed in 0..100 {
        let mut r = request();
        r.seed = seed;
        r.steps = 1;
        let result = simulate(&r).unwrap();
        assert!(result.windows(2).all(|p| p[0].distance > p[1].distance
            || (p[0].distance == p[1].distance && p[0].id < p[1].id)));
    }
}
#[test]
fn fixed_random_vectors() {
    assert_eq!(random_word(0, 0, 0, 0), 0);
    // SplitMix64 finalizer(0x9e3779b97f4a7c15), independently published constant.
    assert_eq!(random_word(0, 0, 0, 1), 0xe220a8397b1dcdaf);
    assert_ne!(random_word(7, 1, 0, 0), random_word(7, 1, 1, 0));
}
