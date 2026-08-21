"""The CLI is the reproducibility boundary: what you type must be what runs."""

import pytest

import config
import main


@pytest.fixture
def clean_config(monkeypatch):
    """Snapshot config so overrides cannot leak between tests."""
    saved = {name: getattr(config, name) for name in main.SETTING_KEYS.values()}
    yield config
    for name, value in saved.items():
        setattr(config, name, value)


class TestOverrides:
    def test_every_flag_reaches_the_config_module(self, clean_config):
        args = main.build_parser().parse_args(
            [
                "--target", "EGFR",
                "--target-chembl-id", "CHEMBL203",
                "--activity-type", "Ki",
                "--min-compounds", "250",
                "--test-size", "0.3",
                "--cv-folds", "10",
                "--random-state", "7",
            ]
        )

        settings = main.apply_overrides(args)

        assert config.TARGET == "EGFR"
        assert config.TARGET_CHEMBL_ID == "CHEMBL203"
        assert config.ACTIVITY_TYPE == "Ki"
        assert config.MIN_COMPOUNDS == 250
        assert config.TEST_SIZE == 0.3
        assert config.CV_FOLDS == 10
        assert config.RANDOM_STATE == 7
        assert settings["TARGET"] == "EGFR"

    def test_defaults_come_from_config(self, clean_config):
        args = main.build_parser().parse_args([])
        settings = main.apply_overrides(args)

        assert settings["TARGET"] == config.TARGET
        assert settings["RANDOM_STATE"] == config.RANDOM_STATE

    def test_settings_dict_covers_every_key(self, clean_config):
        args = main.build_parser().parse_args([])
        settings = main.apply_overrides(args)

        assert set(settings) == set(main.SETTING_KEYS.values())

    def test_y_randomization_can_be_disabled(self, clean_config):
        args = main.build_parser().parse_args(["--no-y-randomization"])
        settings = main.apply_overrides(args)

        assert settings["RUN_Y_RANDOMIZATION"] is False


class TestValidation:
    @pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_impossible_test_size(self, test_size):
        settings = _settings(TEST_SIZE=test_size)

        with pytest.raises(ValueError, match="test-size"):
            main.validate_settings(settings)

    @pytest.mark.parametrize("folds", [0, 1, -3])
    def test_rejects_degenerate_cv(self, folds):
        with pytest.raises(ValueError, match="cv-folds"):
            main.validate_settings(_settings(CV_FOLDS=folds))

    def test_rejects_zero_min_compounds(self):
        with pytest.raises(ValueError, match="min-compounds"):
            main.validate_settings(_settings(MIN_COMPOUNDS=0))

    def test_accepts_a_sane_configuration(self):
        main.validate_settings(_settings())

    def test_y_randomization_runs_must_be_positive_when_enabled(self):
        with pytest.raises(ValueError, match="y-randomization-runs"):
            main.validate_settings(
                _settings(RUN_Y_RANDOMIZATION=True, Y_RANDOMIZATION_RUNS=0)
            )


def _settings(**overrides):
    base = {
        "TEST_SIZE": 0.2,
        "CV_FOLDS": 5,
        "MIN_COMPOUNDS": 1000,
        "RUN_Y_RANDOMIZATION": True,
        "Y_RANDOMIZATION_RUNS": 10,
    }
    base.update(overrides)
    return base
