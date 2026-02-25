from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.utils import Assert


@config_class()
class IntervalConfig(Config):
    # Intervals are a common pattern, so we standardize them with this base class.
    interval: int | None = Field(
        default=None,
        desc="The number of training iterations between each interval. Setting to None will disable.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    offset: int = Field(
        default=0,
        desc="Offset for the first interval.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )

    def _validate(self) -> None:
        if self.interval:
            with self._set_implicit_default(None):
                self.offset %= self.interval
        super()._validate()

    def enabled(self, iteration: int | None = None) -> bool:
        return self.interval and (iteration is None or (iteration - self.offset) % self.interval == 0)

    def is_sub_interval(self, other: "IntervalConfig") -> bool:
        if not self.enabled():
            return True
        elif not other.enabled():
            return False
        return self.interval % other.interval == 0 and (other.offset % other.interval) == (
            self.offset % other.interval
        )

    def assert_sub_interval(self, other: "IntervalConfig") -> None:
        assert self.is_sub_interval(other), f"{self} is not a sub-interval of {other}"

    def get_count(self, iteration) -> int:
        # Number of times this interval was enabled after a given iteration.
        return (iteration - self.offset) // self.interval + 1 if self.enabled() else 0
