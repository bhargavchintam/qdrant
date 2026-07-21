use std::cmp::Reverse;

use ordered_float::Float;

use crate::types::{ScoreType, ScoredPointOffset};

/// TopK implementation following the median algorithm described in
/// <https://quickwit.io/blog/top-k-complexity>.
///
/// Keeps the largest `k` ScoredPointOffset.
#[derive(Default)]
pub struct TopK {
    k: usize,
    elements: Vec<Reverse<ScoredPointOffset>>,
    threshold: ScoreType,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            elements: Vec::with_capacity(2 * k),
            threshold: ScoreType::min_value(),
        }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Returns the minimum score of the top k elements.
    ///
    /// Updated every 2k elements.
    /// Initially set to `ScoreType::MIN`.
    pub fn threshold(&self) -> ScoreType {
        self.threshold
    }

    pub fn push(&mut self, element: ScoredPointOffset) {
        // `k == 0` means no element can ever be kept (see `into_vec`, which always
        // truncates to `self.k`). Bail out immediately instead of accumulating
        // every pushed element: the fill/prune cycle below only fires once
        // `elements.len() == self.k * 2`, which for `k == 0` is 0 - but a push
        // always makes the length at least 1 first, so that check can never be
        // true. Without this guard, `elements` grows without bound for the
        // lifetime of a `TopK::new(0)` instance.
        if self.k == 0 {
            return;
        }
        if element.score > self.threshold {
            self.elements.push(Reverse(element));
            // check if full
            if self.elements.len() == self.k * 2 {
                let (_, median_el, _) = self.elements.select_nth_unstable(self.k - 1);
                self.threshold = median_el.0.score;
                self.elements.truncate(self.k);
            }
        }
    }

    pub fn into_vec(mut self) -> Vec<ScoredPointOffset> {
        self.elements.sort_unstable();
        self.elements.truncate(self.k);
        self.elements.into_iter().map(|Reverse(x)| x).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn empty_with_double_capacity() {
        let top_k = TopK::new(3);
        assert_eq!(top_k.len(), 0);
        assert_eq!(top_k.elements.capacity(), 2 * 3);
        assert_eq!(top_k.threshold(), ScoreType::MIN);
    }

    /// `TopK::new(0)` must never retain elements: without a guard for `k == 0` in
    /// `push`, the fill/prune check (`elements.len() == self.k * 2`) never
    /// triggers, since `k * 2 == 0` but `elements.len()` is always >= 1 right
    /// after a push. That let `elements` grow without bound instead of staying
    /// empty, even though `into_vec()` always truncated the result to 0 anyway -
    /// a client search with `limit == 0` would make the server buffer every
    /// scored candidate instead of discarding them as they arrive.
    #[test]
    fn test_top_k_zero_capacity_stays_bounded() {
        let mut top_k = TopK::new(0);
        for i in 0..1000 {
            top_k.push(ScoredPointOffset {
                score: i as f32,
                idx: i,
            });
        }
        assert_eq!(top_k.len(), 0);
        assert!(top_k.is_empty());
    }

    #[test]
    fn test_top_k_zero_into_vec_is_empty() {
        let mut top_k = TopK::new(0);
        for i in 0..10 {
            top_k.push(ScoredPointOffset {
                score: i as f32,
                idx: i,
            });
        }
        assert_eq!(top_k.into_vec(), Vec::new());
    }

    /// Adjacent boundary case: `k == 1` should keep exactly the single best element.
    #[test]
    fn test_top_k_one_keeps_single_max() {
        let mut top_k = TopK::new(1);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.len(), 1);

        top_k.push(ScoredPointOffset { score: 5.0, idx: 5 });
        // fill/prune triggers at len == 2 * k == 2
        assert_eq!(top_k.threshold(), 5.0);
        assert_eq!(top_k.len(), 1);

        top_k.push(ScoredPointOffset { score: 3.0, idx: 3 });
        assert_eq!(top_k.len(), 1);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], ScoredPointOffset { score: 5.0, idx: 5 });
    }

    #[test]
    fn test_top_k_under() {
        let mut top_k = TopK::new(3);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 1);

        top_k.push(ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 2);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].score, 2.0);
        assert_eq!(res[1].score, 1.0);
    }

    #[test]
    fn test_top_k_over() {
        let mut top_k = TopK::new(3);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.len(), 1);
        assert_eq!(top_k.threshold(), ScoreType::MIN);

        top_k.push(ScoredPointOffset { score: 3.0, idx: 3 });
        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k.threshold(), ScoreType::MIN);

        top_k.push(ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k.threshold(), ScoreType::MIN);

        top_k.push(ScoredPointOffset { score: 4.0, idx: 4 });
        assert_eq!(top_k.len(), 4);
        assert_eq!(top_k.threshold(), ScoreType::MIN);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].score, 4.0);
        assert_eq!(res[1].score, 3.0);
        assert_eq!(res[2].score, 2.0);
    }

    #[test]
    fn test_top_k_pruned() {
        let mut top_k = TopK::new(3);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 1);

        top_k.push(ScoredPointOffset { score: 4.0, idx: 4 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 2);

        top_k.push(ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 3);

        top_k.push(ScoredPointOffset { score: 5.0, idx: 5 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 4);

        top_k.push(ScoredPointOffset { score: 3.0, idx: 3 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 5);

        top_k.push(ScoredPointOffset { score: 6.0, idx: 6 });
        assert_eq!(top_k.threshold(), 4.0);
        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k.elements.capacity(), 6);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].score, 6.0);
        assert_eq!(res[1].score, 5.0);
        assert_eq!(res[2].score, 4.0);
    }

    #[test]
    fn test_top_same_scores() {
        let mut top_k = TopK::new(3);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 1);

        top_k.push(ScoredPointOffset { score: 1.0, idx: 4 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 2);

        top_k.push(ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 3);

        top_k.push(ScoredPointOffset { score: 1.0, idx: 5 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 4);

        top_k.push(ScoredPointOffset { score: 1.0, idx: 3 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 5);

        top_k.push(ScoredPointOffset { score: 1.0, idx: 6 });
        assert_eq!(top_k.threshold(), 1.0);
        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k.elements.capacity(), 6);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0], ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(res[1], ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(res[2], ScoredPointOffset { score: 1.0, idx: 4 });
    }
}
