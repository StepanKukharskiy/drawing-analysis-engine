import unittest

from src.drawing_engine.disciplines.rebar.section_rebar_extraction import _resolve_unique_occurrence_targets


class SectionRebarOccurrenceMatchingTest(unittest.TestCase):
    @staticmethod
    def _candidate(ref, x):
        return {
            "primitive_ref": ref,
            "bbox_display": [x, 10.0, x + 1.0, 50.0],
        }

    def test_overlapping_contacts_close_only_under_unique_injective_matching(self):
        candidates = [
            self._candidate("a.1", 10.0),
            self._candidate("a.2", 12.0),
            self._candidate("b.1", 30.0),
            self._candidate("b.2", 32.0),
        ]
        identities = [
            {"mark_text": "9", "candidate_refs": ["a.1", "a.2"]},
            {"mark_text": "10", "candidate_refs": ["a.1", "a.2", "b.1", "b.2"]},
        ]

        _resolve_unique_occurrence_targets(identities, candidates)

        self.assertEqual(identities[0]["candidate_refs"], ["a.1", "a.2"])
        self.assertEqual(identities[1]["candidate_refs"], ["b.1", "b.2"])
        self.assertEqual(identities[1]["target_assignment_certificate"]["status"], "pass")

    def test_ambiguous_single_callout_is_not_rewritten(self):
        candidates = [
            self._candidate("a.1", 10.0),
            self._candidate("a.2", 12.0),
            self._candidate("b.1", 30.0),
            self._candidate("b.2", 32.0),
        ]
        identities = [
            {"mark_text": "10", "candidate_refs": ["a.1", "a.2", "b.1", "b.2"]},
        ]

        _resolve_unique_occurrence_targets(identities, candidates)

        self.assertEqual(identities[0]["candidate_refs"], ["a.1", "a.2", "b.1", "b.2"])
        self.assertNotIn("target_assignment_certificate", identities[0])


if __name__ == "__main__":
    unittest.main()
