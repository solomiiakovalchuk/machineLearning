import unittest

import numpy as np

from support import PROJECT, control_example, read_json

from ml_project.errors import BatchSizeError, ConfigError, InputValidationError
from ml_project.validation import SchemaValidator


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = read_json(PROJECT / "configs" / "schema.json")
        cls.validator = SchemaValidator(cls.schema)
        cls.names = list(cls.validator.feature_names)
        cls.record = control_example()["record"]

    def rec(self, **changes):
        r = dict(self.record)
        r.update(changes)
        return r

    def test_valid_record_in_schema_order(self):
        out = self.validator.validate_record(self.record)
        self.assertEqual(out.values, tuple(float(self.record[n]) for n in self.names))
        self.assertEqual(out.warnings, ())

    def test_key_order_does_not_matter(self):
        shuffled = dict(reversed(list(self.record.items())))
        self.assertEqual(self.validator.validate_record(shuffled).values,
                         self.validator.validate_record(self.record).values)

    def test_missing_feature(self):
        r = dict(self.record)
        del r["area_mean"]
        with self.assertRaisesRegex(InputValidationError, "missing features: area_mean"):
            self.validator.validate_record(r)

    def test_extra_feature(self):
        with self.assertRaisesRegex(InputValidationError, "unexpected features: id"):
            self.validator.validate_record(self.rec(id=1))

    def test_not_a_mapping(self):
        for bad in ([1.0] * 30, "text", None, 5):
            with self.subTest(bad=bad), self.assertRaises(InputValidationError):
                self.validator.validate_record(bad)

    def test_wrong_types(self):
        bad_values = [True, False, "12.3", "", None, [1.0], {"a": 1}, np.bool_(True), 1 + 2j]
        for bad in bad_values:
            with self.subTest(bad=repr(bad)), self.assertRaises(InputValidationError):
                self.validator.validate_record(self.rec(radius_mean=bad))

    def test_ints_and_numpy_numbers_accepted(self):
        out = self.validator.validate_record(self.rec(radius_mean=15, texture_mean=np.float32(20.5), area_mean=np.int64(800)))
        self.assertEqual(out.values[0], 15.0)

    def test_nan_and_infinity_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf"), np.nan):
            with self.subTest(bad=bad), self.assertRaisesRegex(InputValidationError, "finite"):
                self.validator.validate_record(self.rec(radius_mean=bad))

    def test_huge_int_is_a_controlled_error(self):
        with self.assertRaises(InputValidationError):
            self.validator.validate_record(self.rec(radius_mean=10 ** 400))

    def test_negative_value_blocks(self):
        with self.assertRaisesRegex(InputValidationError, "non-negative"):
            self.validator.validate_record(self.rec(radius_mean=-1.0))

    def test_out_of_range_is_warning_not_error(self):
        out = self.validator.validate_record(self.rec(radius_mean=500.0, texture_mean=0.0))
        self.assertEqual(len(out.warnings), 2)
        self.assertTrue(any(w.startswith("radius_mean=500") for w in out.warnings))
        self.assertTrue(any(w.startswith("texture_mean=0") for w in out.warnings))
        self.assertEqual(out.values[0], 500.0)

    def test_range_boundaries_are_inside(self):
        f = self.schema["features"][0]
        self.assertEqual(self.validator.validate_record(self.rec(radius_mean=f["min"])).warnings, ())
        self.assertEqual(self.validator.validate_record(self.rec(radius_mean=f["max"])).warnings, ())

    def test_batch_limit(self):
        records = [self.record] * 3
        self.assertEqual(len(self.validator.validate_batch(records, 3)), 3)
        with self.assertRaises(BatchSizeError):
            self.validator.validate_batch(records, 2)

    def test_batch_shape_errors(self):
        for bad in ([], "abc", {"a": 1}, None, 5):
            with self.subTest(bad=bad), self.assertRaises(InputValidationError):
                self.validator.validate_batch(bad, 10)

    def test_batch_error_names_the_row(self):
        with self.assertRaisesRegex(InputValidationError, "record 1:"):
            self.validator.validate_batch([self.record, self.rec(radius_mean="x")], 10)

    def test_batch_size_error_is_input_validation_error(self):
        self.assertTrue(issubclass(BatchSizeError, InputValidationError))

    def test_malformed_schema_rejected(self):
        for mutate in (lambda s: s.pop("features"),
                       lambda s: s["features"].append(dict(s["features"][0])),
                       lambda s: s["features"][0].update(dtype="int64"),
                       lambda s: s["features"][0].update(min=9e9)):
            schema = read_json(PROJECT / "configs" / "schema.json")
            mutate(schema)
            with self.subTest(), self.assertRaises(ConfigError):
                SchemaValidator(schema)


if __name__ == "__main__":
    unittest.main()
