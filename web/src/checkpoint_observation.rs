//! Compatibility projection from the current ecosystem tensor to curated policies.

/// Scalar prefix encoded by the curated checkpoints.
const CHECKPOINT_PROPRIOCEPTION_SIZE: usize = 21;

/// Compact scalar prefix emitted by the current runtime.
const CURRENT_PROPRIOCEPTION_SIZE: usize = 11;

/// Fixed semantic rays shared by the curated checkpoints and current runtime.
const RAY_COUNT: usize = 24;

/// Distance, presence, and eight semantic channels retained by curated policies.
const CHECKPOINT_RAY_WIDTH: usize = 10;

/// Distance and ten semantic channels emitted by the current runtime.
const CURRENT_RAY_WIDTH: usize = 11;

/// Number of semantic one-hot channels emitted by the current runtime.
const CURRENT_RAY_KIND_COUNT: usize = 10;

/// Curriculum lesson slots retained by the curated six-stage policies.
const CHECKPOINT_LESSON_COUNT: usize = 6;

/// Actor input width encoded by every curated checkpoint in the release manifest.
pub(super) const CHECKPOINT_OBSERVATION_SIZE: usize =
    CHECKPOINT_PROPRIOCEPTION_SIZE + RAY_COUNT * CHECKPOINT_RAY_WIDTH + CHECKPOINT_LESSON_COUNT;

/// Observation width emitted by the current ecosystem runtime.
pub(super) const CURRENT_OBSERVATION_SIZE: usize =
    CURRENT_PROPRIOCEPTION_SIZE + RAY_COUNT * CURRENT_RAY_WIDTH;

/// Project one current observation into the exact six-stage checkpoint schema.
pub(super) fn project(
    current: &[f32; CURRENT_OBSERVATION_SIZE],
) -> [f32; CHECKPOINT_OBSERVATION_SIZE] {
    let mut checkpoint = [0.0; CHECKPOINT_OBSERVATION_SIZE];

    // The first seven physiology and movement features retain their original order.
    for index in 0..7 {
        if let (Some(source), Some(target)) = (current.get(index), checkpoint.get_mut(index)) {
            *target = *source;
        }
    }
    for (checkpoint_index, current_index) in [(9, 7), (18, 8), (19, 9), (20, 10)] {
        if let Some(value) = current.get(current_index).copied() {
            write_feature(&mut checkpoint, checkpoint_index, value);
        }
    }

    // Reinsert the presence channel and omit the newer Gorge and Bridge kinds.
    for ray_index in 0..RAY_COUNT {
        let current_start = CURRENT_PROPRIOCEPTION_SIZE + ray_index * CURRENT_RAY_WIDTH;
        let distance = current.get(current_start).copied().unwrap_or_default();
        let kind = (0..CURRENT_RAY_KIND_COUNT).find(|kind_index| {
            current
                .get(current_start + 1 + kind_index)
                .is_some_and(|value| *value > 0.5)
        });
        let Some(kind_index) = kind else {
            continue;
        };
        if kind_index < 8 {
            let checkpoint_start =
                CHECKPOINT_PROPRIOCEPTION_SIZE + ray_index * CHECKPOINT_RAY_WIDTH;
            write_feature(&mut checkpoint, checkpoint_start, distance);
            write_feature(&mut checkpoint, checkpoint_start + 1, 1.0);
            write_feature(&mut checkpoint, checkpoint_start + 2 + kind_index, 1.0);
        }
    }

    checkpoint
}

/// Write one compatibility feature when its fixed schema index is present.
fn write_feature(checkpoint: &mut [f32; CHECKPOINT_OBSERVATION_SIZE], index: usize, value: f32) {
    if let Some(feature) = checkpoint.get_mut(index) {
        *feature = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Projection leaves removed species, resource-summary, and lesson slots zero.
    #[test]
    fn current_observation_projects_to_curated_checkpoint_schema() {
        let mut current = [0.0; CURRENT_OBSERVATION_SIZE];
        for index in 0..CURRENT_PROPRIOCEPTION_SIZE {
            if let Some(feature) = current.get_mut(index) {
                *feature = index as f32 / 10.0;
            }
        }
        let first_ray = CURRENT_PROPRIOCEPTION_SIZE;
        if let Some(distance) = current.get_mut(first_ray) {
            *distance = 0.25;
        }
        if let Some(food) = current.get_mut(first_ray + 1) {
            *food = 1.0;
        }

        let checkpoint = project(&current);

        assert_eq!(checkpoint.len(), CHECKPOINT_OBSERVATION_SIZE);
        assert_eq!(checkpoint.first(), current.first());
        assert_eq!(checkpoint.get(7), Some(&0.0));
        assert_eq!(checkpoint.get(8), Some(&0.0));
        assert_eq!(checkpoint.get(9), current.get(7));
        assert_eq!(checkpoint.get(10), Some(&0.0));
        assert_eq!(checkpoint.get(11), Some(&0.0));
        assert_eq!(checkpoint.get(18), current.get(8));
        let checkpoint_ray = CHECKPOINT_PROPRIOCEPTION_SIZE;
        assert_eq!(checkpoint.get(checkpoint_ray), Some(&0.25));
        assert_eq!(checkpoint.get(checkpoint_ray + 1), Some(&1.0));
        assert_eq!(checkpoint.get(checkpoint_ray + 2), Some(&1.0));
        let lessons = checkpoint
            .iter()
            .rev()
            .take(CHECKPOINT_LESSON_COUNT)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(lessons, vec![0.0; CHECKPOINT_LESSON_COUNT]);
    }
}
