# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

from pydantic import BaseModel, field_validator


class HEConfig(BaseModel):
    """Configuration for a CKKS homomorphic encryption context.

    Attributes:
        batch_size: Number of slots per ciphertext. Must be a power of 2, at most 2^14.
        mult_depth: Maximum multiplicative depth (number of sequential multiplications allowed).
        scale_factor_bits: Approximate number of bits of precision in the scale factor.
        bootstrapping: Whether to set up bootstrapping keys and parameters.
    """

    batch_size: int
    mult_depth: int = 10
    scale_factor_bits: int = 40
    bootstrapping: bool = False

    @field_validator("batch_size")
    @classmethod
    def batch_size_must_be_power_of_two(cls, v: int) -> int:
        if v <= 0 or v & (v - 1) != 0:
            raise ValueError(f"batch_size must be a positive power of 2, got {v}")
        return v

    @field_validator("mult_depth")
    @classmethod
    def mult_depth_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"mult_depth must be positive, got {v}")
        return v

    @field_validator("scale_factor_bits")
    @classmethod
    def scale_factor_bits_in_range(cls, v: int) -> int:
        if not (10 <= v <= 60):
            raise ValueError(f"scale_factor_bits must be between 10 and 60, got {v}")
        return v

    @property
    def serialization_key(self) -> str:
        """Unique string identifier for this configuration, used for serialization paths."""
        return f"{self.batch_size}-{self.mult_depth}-{self.scale_factor_bits}-{int(self.bootstrapping)}"
