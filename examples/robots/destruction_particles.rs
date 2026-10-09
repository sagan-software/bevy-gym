//! Shared meshes and bounded smoke, flash, and spark lifetimes.

use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    light::NotShadowCaster,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use std::time::Duration;

/// Presentation particles, removed on expiry or episode reset.
#[derive(Component)]
pub(super) struct Particle {
    /// Time since emission, independent of the flight task's paused state.
    age: Duration,
    /// Maximum visible lifetime.
    lifetime: Duration,
    /// World velocity in metres per second.
    velocity: Vec3,
    /// Distinct motion and appearance recipes.
    kind: Kind,
    /// Initial visual radius in metres.
    radius: f32,
}

/// Visual particle recipes; no values flow into training observations.
#[derive(Clone, Copy)]
pub(super) enum Kind {
    /// Short bright sphere at the impact.
    Flash,
    /// Small outward particle accelerated by gravity.
    Spark,
    /// Rising puff with increasing size and fading opacity.
    Smoke,
}

/// One set of reusable assets for every destruction event.
#[derive(Resource)]
pub(super) struct Materials {
    /// Low-resolution particle sphere shared by all emitters.
    sphere: Handle<Mesh>,
    /// Camera-facing square carrying a soft radial opacity texture.
    billboard: Handle<Mesh>,
    /// Fragment box dimensions match the debris colliders.
    pub(super) fragment: Handle<Mesh>,
    /// Soft camera-facing flash material, separate from the small spark spheres.
    flash: Handle<StandardMaterial>,
    /// Hot emission for spark spheres.
    fire: Handle<StandardMaterial>,
    /// Four fixed smoke opacity levels avoid allocating materials each frame.
    smoke: [Handle<StandardMaterial>; 4],
    /// Wreckage material shared by every fragment.
    pub(super) steel: Handle<StandardMaterial>,
}

/// Create fixed assets once; repeated resets do not grow the asset stores.
pub(super) fn setup(
    mut commands: Commands<'_, '_>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
    mut images: ResMut<'_, Assets<Image>>,
) {
    let opacity = images.add(radial_texture());
    let smoke = [0.35, 0.22, 0.12, 0.04].map(|alpha| {
        materials.add(StandardMaterial {
            base_color: Color::srgba(0.12, 0.14, 0.15, alpha),
            base_color_texture: Some(opacity.clone()),
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            unlit: true,
            ..default()
        })
    });
    commands.insert_resource(Materials {
        sphere: meshes.add(Sphere::new(1.0).mesh().uv(8, 6)),
        billboard: meshes.add(Rectangle::new(2.0, 2.0)),
        fragment: meshes.add(Cuboid::new(0.22, 0.08, 0.22)),
        flash: materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 0.92, 0.7, 0.95),
            base_color_texture: Some(opacity),
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            unlit: true,
            ..default()
        }),
        fire: materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.62, 0.12),
            emissive: LinearRgba::new(5.0, 1.6, 0.1, 1.0),
            unlit: true,
            ..default()
        }),
        smoke,
        steel: materials.add(Color::srgb(0.12, 0.14, 0.15)),
    });
}

impl Materials {
    /// Emit a short flash and twelve outward sparks at one destruction point.
    pub(super) fn burst(&self, commands: &mut Commands<'_, '_>, position: Vec3, size: f32) {
        self.spawn(
            commands,
            position,
            Vec3::ZERO,
            Kind::Flash,
            Duration::from_millis(160),
            size,
        );
        for index in 0..12 {
            let angle = index as f32 * std::f32::consts::TAU / 12.0;
            let velocity = Vec3::new(
                angle.cos(),
                ((index % 3) as f32).mul_add(0.4, 0.6),
                angle.sin(),
            ) * 2.5;
            self.spawn(
                commands,
                position,
                velocity,
                Kind::Spark,
                Duration::from_millis(650),
                0.025,
            );
        }
    }

    /// Emit one rising smoke puff; the caller controls the bounded emission period.
    pub(super) fn puff(&self, commands: &mut Commands<'_, '_>, position: Vec3) {
        self.spawn(
            commands,
            position,
            Vec3::new(0.12, 0.65, 0.08),
            Kind::Smoke,
            Duration::from_secs(2),
            0.07,
        );
    }

    /// Retain only per-particle motion and lifetime; all GPU assets are shared.
    pub(super) fn spawn(
        &self,
        commands: &mut Commands<'_, '_>,
        position: Vec3,
        velocity: Vec3,
        kind: Kind,
        lifetime: Duration,
        size: f32,
    ) {
        let material = match kind {
            Kind::Smoke => self.smoke[0].clone(),
            Kind::Flash => self.flash.clone(),
            Kind::Spark => self.fire.clone(),
        };
        commands.spawn((
            Particle {
                age: Duration::ZERO,
                lifetime,
                velocity,
                kind,
                radius: size,
            },
            Mesh3d(match kind {
                Kind::Smoke | Kind::Flash => self.billboard.clone(),
                Kind::Spark => self.sphere.clone(),
            }),
            NotShadowCaster,
            MeshMaterial3d(material),
            Transform::from_translation(position).with_scale(Vec3::splat(size)),
        ));
    }
}

/// Advance feedback even after a crash stops the flight environment.
pub(super) fn animate(
    mut commands: Commands<'_, '_>,
    time: Res<'_, Time>,
    materials: Res<'_, Materials>,
    camera: Query<'_, '_, &Transform, (With<Camera3d>, Without<Particle>)>,
    mut particles: Query<
        '_,
        '_,
        (
            Entity,
            &mut Particle,
            &mut Transform,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
    >,
) {
    let delta = time.delta().min(Duration::from_millis(100));
    let camera_position = camera.iter().next().map(|transform| transform.translation);
    for (entity, mut particle, mut transform, mut material) in &mut particles {
        particle.age += delta;
        if particle.age >= particle.lifetime {
            commands.entity(entity).despawn();
            continue;
        }
        let age = particle.age.as_secs_f32();
        let fraction = age / particle.lifetime.as_secs_f32();
        // Soft quads face the camera; spark spheres need no orientation update.
        if matches!(particle.kind, Kind::Smoke | Kind::Flash) {
            if let Some(position) = camera_position {
                transform.look_at(position, Vec3::Y);
            }
        }
        match particle.kind {
            Kind::Spark => {
                particle.velocity.y = 9.81_f32.mul_add(-delta.as_secs_f32(), particle.velocity.y);
            }
            Kind::Smoke => {
                transform.scale = Vec3::splat(age.mul_add(0.13, particle.radius));
                let [dense, medium, thin, faint] = &materials.smoke;
                material.0 = if fraction < 0.25 {
                    dense
                } else if fraction < 0.5 {
                    medium
                } else if fraction < 0.75 {
                    thin
                } else {
                    faint
                }
                .clone();
            }
            Kind::Flash => {
                transform.scale =
                    Vec3::splat(particle.radius * (1.0 + fraction) * (1.0 - fraction));
            }
        }
        transform.translation += particle.velocity * delta.as_secs_f32();
    }
}

/// Generate one 33-pixel radial alpha mask; the square's edges are fully transparent.
fn radial_texture() -> Image {
    let mut pixels = Vec::with_capacity(33 * 33 * 4);
    for y in -16_i32..=16 {
        for x in -16_i32..=16 {
            // Squared pixel radius is 256. Squaring its remaining fraction softens the edge.
            let remaining = (256 - x * x - y * y).max(0);
            let alpha = u8::try_from(remaining * remaining * 255 / 65_536)
                .expect("radial opacity lies in 0..=255");
            pixels.extend([255, 255, 255, alpha]);
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: 33,
            height: 33,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::linear();
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_mask_has_an_opaque_centre_and_transparent_edges() {
        let image = radial_texture();
        let bytes = image.data.unwrap();
        assert_eq!(bytes.len(), 33 * 33 * 4);
        assert_eq!(
            &bytes[(16 * 33 + 16) * 4..(16 * 33 + 17) * 4],
            &[255, 255, 255, 255]
        );
        for index in 0..33 {
            for pixel in [index, 32 * 33 + index, index * 33, index * 33 + 32] {
                assert_eq!(bytes[pixel * 4 + 3], 0);
            }
        }
        let intermediate = bytes[(16 * 33 + 24) * 4 + 3];
        assert!(intermediate > 0 && intermediate < 255);
    }
    #[test]
    fn flash_uses_a_soft_billboard_and_faces_the_camera() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<Image>()
            .init_asset::<StandardMaterial>()
            .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                Duration::from_millis(10),
            ))
            .add_systems(Startup, setup)
            .add_systems(Update, animate);
        let eye = Vec3::new(3.0, 3.0, 3.0);
        app.world_mut()
            .spawn((Camera3d::default(), Transform::from_translation(eye)));
        app.update();
        app.world_mut()
            .resource_scope(|world, materials: Mut<'_, Materials>| {
                materials.burst(&mut world.commands(), Vec3::ZERO, 0.6);
            });
        app.world_mut().flush();
        app.update();
        let world = app.world_mut();
        let billboard = world.resource::<Materials>().billboard.clone();
        let (mesh, material, pose) = world
            .query::<(
                &Particle,
                &Mesh3d,
                &MeshMaterial3d<StandardMaterial>,
                &Transform,
            )>()
            .iter(world)
            .find_map(|(particle, mesh, material, pose)| {
                matches!(particle.kind, Kind::Flash).then_some((mesh, material, pose))
            })
            .expect("Live flash");
        assert_eq!(mesh.0, billboard);
        let material = world
            .resource::<Assets<StandardMaterial>>()
            .get(&material.0)
            .expect("Flash material");
        assert!(material.base_color_texture.is_some());
        assert_eq!(material.alpha_mode, AlphaMode::Blend);
        assert!(pose.forward().dot(eye.normalize()) > 0.999);
    }
}
