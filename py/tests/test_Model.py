from Model import Hyperparameters
import pytest


class TestHyperparametersConstructor:
    """Test the constructor method for Hyperparameters."""

    @pytest.mark.parametrize(
        "model_name, model_code, params, expected_out",
        [
            ####
            #### Testing model_name ####
            ####
            ## empty string in name
            ("", "valid_code", {}, ValueError),
            ## None should be invalid
            (None, "valid_code", {}, TypeError),
            ## None is Falsey - so try True to test boolean
            (True, "valid_code", {}, TypeError),
            ## A list of strings should be invalid
            (["name"], "valid_code", {}, TypeError),
            ## A dict with String value should be invalid
            ({"key": "value"}, "valid_code", {}, TypeError),
            ####
            #### Testing model_code ####
            ####
            ("valid_name", "", {}, ValueError),
            ("valid_name", None, {}, TypeError),
            ("valid_name", True, {}, TypeError),
            ("valid_name", ["name"], {}, TypeError),
            ("valid_name", {"key": "value"}, {}, TypeError),
            ####
            #### Testing params ####
            ####
            ("valid_name", "valid_code", None, TypeError),
            ("valid_name", "valid_code", True, TypeError),
            ## a set should be invalid
            ("valid_name", "valid_code", {"a"}, TypeError),
            ## a string should be invalid
            ("valid_name", "valid_code", "", TypeError),
            ## ParameterGrid from sci-kit Learn raises a TypeError is a single value is not in a list
            ("valid_name", "valid_code", {"key", "a"}, TypeError),
            ("valid_name", "valid_code", {"key", 2}, TypeError),
        ],
    )
    def test_input_validation(self, model_name, model_code, params, expected_out):
        ## test empty string
        with pytest.raises(expected_out):
            ## assume the params are an empty dictionary
            Hyperparameters(model_name=model_name, model_code=model_code, params=params)

    def test_model_code(self):
        """Model code should turn the string into upper case"""

        ## ensure the lower case are made upper and the upper remain unchanged
        h = Hyperparameters(model_name="valid_name", model_code="lower_CASE", params={})
        code = h.model_code

        ## check the length is correct
        assert len(set([code, code.upper()])) == len(set([code]))

        ## check the value is expected - input was lower_CASE
        assert code == code.upper()


class TestHyperparametersParamMethods:
    """Test the dictionary or list of dictionaries are correctly handled by the methods.
    Focus on Hyperparameters.create_grid() and Hyperparameters.create_ids()"""

    @pytest.mark.parametrize(
        "params, expected_out",
        [
            ## check the empty dict is converted to [{NA:NA}]
            ({}, [{"NA": "NA"}]),
            ## check a single string or int are not put into a list (inside the dictionary)
            ({"key": ["value"]}, [{"key": "value"}]),
            ({"key": [1.0]}, [{"key": 1.0}]),
            ## check a list of values return a list of dictionaries
            (
                {"key": ["a", "b", "c"]},
                [{"key": "a"}, {"key": "b"}, {"key": "c"}],
            ),
            ({"key": [1, 2]}, [{"key": 1}, {"key": 2}]),
            ## check multiple dictionaries are combined
            (
                {"a": [1, 2], "b": ["x", "y"]},
                [
                    {"a": 1, "b": "x"},
                    {"a": 1, "b": "y"},
                    {"a": 2, "b": "x"},
                    {"a": 2, "b": "y"},
                ],
            ),
        ],
    )
    def test_create_grid(self, params, expected_out):
        h = Hyperparameters(
            model_name="valid_name", model_code="valid_code", params=params
        )

        assert h.create_grid() == expected_out
