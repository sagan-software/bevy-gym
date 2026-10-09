//! Find conservative horizontal routes through the authored blockout.
//!
//! Rapier 0.36 casts use a unit time interval and displacement in metres:
//! <https://docs.rs/rapier3d/0.36.0/rapier3d/pipeline/struct.PhysicsWorld.html#method.cast_shape>.

use crate::arena::layout::{self, Block};
use bevy::math::Vec3;
use rapier3d::{
    parry::query::ShapeCastOptions,
    prelude::{
        ColliderBuilder, PhysicsWorld, Pose, QueryFilter, Ray, RigidBodyBuilder, Shape,
        SharedShape, Vector,
    },
};

/// Number of cells per horizontal grid axis.
const SIDE: usize = 39;

/// An index created only while constructing or enumerating this fixed graph.
#[derive(Clone, Copy, PartialEq, Eq)]
struct NodeId(usize);

/// A fixed-height flight cell and its collision-tested neighbours.
struct Cell {
    /// World centre in metres, at altitude two metres.
    point: Vec3,
    /// Whether the normal clearance sphere fits at the centre.
    free: bool,
    /// At most four cardinal neighbours with a clear sphere sweep.
    edges: Vec<NodeId>,
}

/// Reused breadth-first search storage, bounded by the fixed grid size.
struct Search {
    /// First predecessor; the root points to itself.
    previous: Vec<Option<NodeId>>,
    /// Shortest path length from the measured position, in metres.
    distances: Vec<f32>,
    /// Reachable nodes in deterministic breadth-first order.
    queue: Vec<NodeId>,
    /// Selected route stored from the destination back to the root.
    route: Vec<NodeId>,
}

/// Query-only world with conservative horizontal clearance and reusable scratch space.
pub(super) struct Map {
    /// Static authored boxes; no moving actor is inserted.
    world: PhysicsWorld,
    /// Normal clearance radius of 0.9 metres.
    sphere: SharedShape,
    /// Escape radius of 0.45 metres encloses the physical body at any orientation.
    body: SharedShape,
    /// The fixed graph's cells and tested neighbour connections.
    cells: Vec<Cell>,
    /// Potential target samples in free space, independent of the character.
    pub(super) targets: Vec<Vec3>,
    /// Buffers reused on each navigation decision.
    search: Search,
}

impl Default for Map {
    fn default() -> Self {
        Self::new(&layout::blocks())
    }
}

impl Map {
    /// Build a query-only map from the exact boxes used for rendering and collision.
    fn new(blocks: &[Block]) -> Self {
        let mut world = PhysicsWorld::default();
        for block in blocks {
            world.insert(
                RigidBodyBuilder::fixed()
                    .translation(Vector::from_array(block.centre.to_array()))
                    .rotation(Vector::from_array(
                        block.rotation.to_scaled_axis().to_array(),
                    )),
                ColliderBuilder::cuboid(block.half.x, block.half.y, block.half.z),
            );
        }
        world.step();
        let cells = (0..SIDE * SIDE)
            .map(|index| Cell {
                point: Vec3::new(
                    ((index % SIDE) as f32).mul_add(0.5, -9.5),
                    2.0,
                    ((index / SIDE) as f32).mul_add(0.5, -9.5),
                ),
                free: false,
                edges: Vec::with_capacity(4),
            })
            .collect();
        let mut map = Self {
            world,
            sphere: SharedShape::ball(0.9),
            body: SharedShape::ball(0.45),
            cells,
            targets: Vec::new(),
            search: Search {
                previous: vec![None; SIDE * SIDE],
                distances: vec![f32::INFINITY; SIDE * SIDE],
                queue: Vec::with_capacity(SIDE * SIDE),
                route: Vec::with_capacity(SIDE * SIDE),
            },
        };
        map.build_targets();
        map.build_edges();
        map
    }

    /// Sample possible target points, excluding locations inside authored solids.
    fn build_targets(&mut self) {
        let shape = SharedShape::ball(0.1);
        for x in -9..=9 {
            for z in -9..=9 {
                let point = Vec3::new(x as f32, 1.0, z as f32);
                let pose = Pose::from_translation(Vector::from_array(point.to_array()));
                if self
                    .world
                    .intersect_shape(pose, shape.as_ref(), QueryFilter::default())
                    .next()
                    .is_none()
                {
                    self.targets.push(point);
                }
            }
        }
    }

    /// Connect cardinal neighbours only when both occupancy and the full sweep are clear.
    fn build_edges(&mut self) {
        for index in 0..self.cells.len() {
            let point = self.cell(NodeId(index)).point;
            let free = self.clear(point, point);
            self.cells
                .get_mut(index)
                .expect("Enumerated graph cell")
                .free = free;
        }
        for index in 0..self.cells.len() {
            let node = NodeId(index);
            if !self.cell(node).free {
                continue;
            }
            for next in neighbours(node).into_iter().flatten() {
                if self.cell(next).free && self.clear(self.cell(node).point, self.cell(next).point)
                {
                    self.cells
                        .get_mut(index)
                        .expect("Enumerated graph cell")
                        .edges
                        .push(next);
                }
            }
        }
    }

    /// A node identifier always originates from this map's fixed grid.
    fn cell(&self, node: NodeId) -> &Cell {
        self.cells
            .get(node.0)
            .expect("Node belongs to the fixed grid")
    }

    /// Sweep the normal clearance sphere through the whole requested displacement.
    fn clear(&self, from: Vec3, to: Vec3) -> bool {
        self.clear_shape(from, to, self.sphere.as_ref())
    }

    /// Reject starting penetration before casting; cast time one means the full segment.
    fn clear_shape(&self, from: Vec3, to: Vec3, shape: &dyn Shape) -> bool {
        if !from.is_finite() || !to.is_finite() {
            return false;
        }
        let pose = Pose::from_translation(Vector::from_array(from.to_array()));
        if self
            .world
            .intersect_shape(pose, shape, QueryFilter::default())
            .next()
            .is_some()
        {
            return false;
        }
        let delta = to - from;
        if delta.length_squared() < 1e-10 {
            return true;
        }
        self.world
            .cast_shape(
                &pose,
                Vector::from_array(delta.to_array()),
                shape,
                ShapeCastOptions {
                    max_time_of_impact: 1.0,
                    stop_at_penetration: true,
                    ..Default::default()
                },
                QueryFilter::default(),
            )
            .is_none()
    }

    /// A solid ray includes the endpoint and rejects undefined directions.
    pub(super) fn visible(&self, from: Vec3, to: Vec3) -> bool {
        let delta = to - from;
        let distance = delta.length();
        if !distance.is_finite() || distance <= f32::EPSILON {
            return false;
        }
        self.world
            .cast_ray(
                &Ray::new(
                    Vector::from_array(from.to_array()),
                    Vector::from_array((delta / distance).to_array()),
                ),
                distance,
                true,
                QueryFilter::default(),
            )
            .is_none()
    }

    /// Connect the actual position to the nearest reachable free cell.
    fn root(&self, from: Vec3, escaping: bool) -> Option<(NodeId, f32)> {
        let mut root = None;
        let mut nearest = f32::INFINITY;
        for (index, cell) in self.cells.iter().enumerate() {
            let distance = cell.point.distance_squared(from);
            if cell.free
                && distance < nearest
                && (self.clear(from, cell.point)
                    || (escaping && self.clear_shape(from, cell.point, self.body.as_ref())))
            {
                root = Some(NodeId(index));
                nearest = distance;
            }
        }
        root.map(|node| (node, nearest.sqrt()))
    }

    /// Fill shortest paths in O(cells + edges) time with fixed-size retained storage.
    fn explore(&mut self, root: NodeId, distance: f32) {
        self.search.previous.fill(None);
        self.search.distances.fill(f32::INFINITY);
        self.search.queue.clear();
        self.search.queue.push(root);
        *self
            .search
            .previous
            .get_mut(root.0)
            .expect("Root belongs to grid") = Some(root);
        *self
            .search
            .distances
            .get_mut(root.0)
            .expect("Root belongs to grid") = distance;
        let mut head = 0;
        while let Some(&node) = self.search.queue.get(head) {
            head += 1;
            let cell = self.cells.get(node.0).expect("Queued grid cell");
            let distance = *self.search.distances.get(node.0).expect("Queued distance") + 0.5;
            for &next in &cell.edges {
                let previous = self
                    .search
                    .previous
                    .get_mut(next.0)
                    .expect("Neighbour belongs to grid");
                if previous.is_none() {
                    *previous = Some(node);
                    *self
                        .search
                        .distances
                        .get_mut(next.0)
                        .expect("Neighbour distance") = distance;
                    self.search.queue.push(next);
                }
            }
        }
    }

    /// Minimize path length plus twice the error from a six-metre viewing distance.
    fn destination(&self, target: Vec3) -> Option<NodeId> {
        let mut best = None;
        let mut score = f32::INFINITY;
        for &node in &self.search.queue {
            let point = self.cell(node).point;
            let range = point.distance(target);
            let distance = *self.search.distances.get(node.0)?;
            let candidate = (range - 6.0).abs().mul_add(2.0, distance);
            if (2.0..=12.0).contains(&range) && candidate < score && self.visible(point, target) {
                best = Some(node);
                score = candidate;
            }
        }
        best
    }

    /// Pick a reachable viewing cell, then a clear shortcut no longer than 2.5 metres.
    pub(super) fn viewpoint(&mut self, from: Vec3, target: Vec3) -> Option<Vec3> {
        let escaping = !self.clear(from, from);
        let (root, distance) = self.root(from, escaping)?;
        if escaping {
            return Some(self.cell(root).point);
        }
        self.explore(root, distance);
        let mut current = self.destination(target)?;
        self.search.route.clear();
        self.search.route.push(current);
        while current != root {
            current = self.search.previous.get(current.0).copied().flatten()?;
            self.search.route.push(current);
        }
        let mut next = self.cell(root).point;
        for &node in self.search.route.iter().rev() {
            let point = self.cell(node).point;
            if point.distance(from) <= 2.5 && self.clear(from, point) {
                next = point;
            }
        }
        Some(next)
    }
}

/// Cardinal graph neighbours preserve left, right, back, front tie-breaking order.
fn neighbours(node: NodeId) -> [Option<NodeId>; 4] {
    let x = node.0 % SIDE;
    let z = node.0 / SIDE;
    [
        x.checked_sub(1).map(|x| NodeId(z * SIDE + x)),
        (x + 1 < SIDE).then_some(NodeId(z * SIDE + x + 1)),
        z.checked_sub(1).map(|z| NodeId(z * SIDE + x)),
        (z + 1 < SIDE).then_some(NodeId((z + 1) * SIDE + x)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Quat;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn a_closed_door_requires_a_view_through_the_house_window() {
        let mut blocks = layout::blocks();
        blocks.push(Block {
            centre: Vec3::new(-2.0, 1.2, -3.0),
            half: Vec3::new(0.15, 1.2, 1.0),
            rotation: Quat::IDENTITY,
            surface: layout::Surface::Wall,
        });
        let mut map = Map::new(&blocks);
        let target = Vec3::new(-5.0, 1.5, -3.0);
        let mut position = Vec3::new(0.0, 2.0, 3.0);
        assert!(!map.visible(position, target));
        for _ in 0..20 {
            let next = map.viewpoint(position, target).expect("Window route");
            assert!(map.clear(position, next));
            assert!(position.distance(next) <= 2.5);
            position = next;
        }
        assert!(map.visible(position, target));
        assert!(position.z > 0.15);
        let crossing = position.lerp(target, position.z / (position.z - target.z));
        assert!((-6.0..=-4.0).contains(&crossing.x));
        assert!((1.0..=2.5).contains(&crossing.y));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn clearance_margin_can_escape_but_a_penetrating_body_cannot() {
        let mut map = Map::default();
        let from = Vec3::new(0.7061, 1.9957, 6.2862);
        let target = Vec3::new(5.0, 1.5, 1.0);
        assert!(!map.clear(from, from));
        assert!(map.clear_shape(from, from, map.body.as_ref()));
        let next = map.viewpoint(from, target).expect("Clear escape cell");
        assert!(map.clear_shape(from, next, map.body.as_ref()));
        assert!(map.clear(next, next));
        assert!(map.viewpoint(Vec3::new(-8.0, 2.0, -3.0), target).is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn invalid_queries_and_unreachable_targets_have_no_route() {
        let mut map = Map::default();
        let point = Vec3::new(0.0, 2.0, 0.0);
        for invalid in [Vec3::NAN, Vec3::splat(f32::INFINITY)] {
            assert!(!map.clear(point, invalid));
            assert!(!map.clear(invalid, point));
            assert!(!map.visible(point, invalid));
            assert!(!map.visible(invalid, point));
            assert!(map.viewpoint(invalid, point).is_none());
            assert!(map.viewpoint(point, invalid).is_none());
        }
        assert!(map.clear(point, point));
        assert!(!map.visible(point, point));
        assert!(map.viewpoint(point, Vec3::new(100.0, 2.0, 100.0)).is_none());
    }
}
