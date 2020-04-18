from gym.envs.registration import register

register(
    id="projectile-v0",
    entry_point="gym_projectile.envs:Projectile_v0",
)
