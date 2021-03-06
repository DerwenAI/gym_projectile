# gym_projectile

A simple implementation of an [OpenAI Gym](http://gym.openai.com/)
environment, to show a simulation of a projectile trajectory.

Assumes: level ground, no atmosphere


## Usage

Clone the repo and connect into its top level directory.

Then to run the `gym` example:

```
pip install -r requirements.txt
pip install -e gym-projectile

python example.py
```

Then to run Ray RLlib to train a policy based on this environment:

```
python train.py
```


## Kudos

h/t:

  - <https://github.com/apoddar573/Tic-Tac-Toe-Gym_Environment/>
  - <https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa>
  - <https://github.com/openai/gym/blob/master/docs/creating-environments.md>
  - <https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai>
