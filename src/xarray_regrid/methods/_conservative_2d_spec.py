from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import Literal

# Geometry backend used to build cell polygons and compute their intersections.
# - "planar": raw shapely on the user's coords
# - "cea": Lambert cylindrical equal-area projection of 1D lat/lon (degrees)
#          — analytic spherical areas at planar cost
# - "s2": true great-circle polygons on the sphere via the optional `spherely`
#         package (s2geometry). Each manifold is a GeometryBackend (see the
#         _BACKENDS registry in conservative_2d).
Manifold = Literal["planar", "cea", "s2"]


@dataclass(frozen=True)
class RegridSpec:
    """Canonical metadata describing a source->target regridding layout."""

    src_dims: tuple[Hashable, ...]
    dst_dims: tuple[Hashable, ...]
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    x_coord: str
    y_coord: str
    manifold: Manifold

    def transposed(self) -> "RegridSpec":
        """Swap source and target — same layout, opposite direction."""
        return replace(
            self,
            src_dims=self.dst_dims,
            dst_dims=self.src_dims,
            src_shape=self.dst_shape,
            dst_shape=self.src_shape,
        )
