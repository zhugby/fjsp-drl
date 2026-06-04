from gym.envs.registration import register
import gym

# Registrar for the gym environment
# https://www.gymlibrary.ml/content/environment_creation/ for reference
register(
    id='fjsp-v0',  # Environment name (including version number)
    entry_point='env.fjsp_env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)

def make_fjsp_env(case, env_paras, data_source='case'):
    '''
    Create the FJSP environment while keeping compatibility with newer Gym versions.
    Gym 0.26 enables PassiveEnvChecker by default, but this project uses a legacy
    tensor-based environment API rather than Gym spaces.
    '''
    try:
        return gym.make('fjsp-v0', case=case, env_paras=env_paras, data_source=data_source, disable_env_checker=True)
    except TypeError as exc:
        if 'disable_env_checker' not in str(exc):
            raise
        return gym.make('fjsp-v0', case=case, env_paras=env_paras, data_source=data_source)
