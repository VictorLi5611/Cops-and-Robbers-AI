[![Contributors][contributors-shield1]][contributors-url1]
[![Contributors][contributors-shield2]][contributors-url2]
[![Contributors][contributors-shield3]][contributors-url3]
[![Contributors][contributors-shield4]][contributors-url4]
[![Contributors][contributors-shield5]][contributors-url5]


<p align="center">
  <h2 align="center">COMP 4900C Team Project (Fall 2023)</h2>
  <h3 align="center">Reinforcement Learning for Cops and Robbers</h3>
<br>

<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>

## About The Project

This project was created by Victor Li, Sebastian Beimers, Laura Jin, Shawn Sun and Evan Li for Carleton University's COMP 4900C Fall 2023 course, taught by JunWen.

This Project is a Reinforced learning group project on the optimal path for the Robber to get to the Vault by dodging cops that have a set patrol path.


## Getting Started
To get a local copy of this project up and running, please follow the steps below.

### Installation

1. Set up a virtual environment using `python3 -m venv venv` for mac or `py -m venv venv` for windows


2. Activate the virtual environment using `source venv/bin/activate` for mac or `Set-ExecutionPolicy Unrestricted -Scope Process` then `venv\Scripts\activate` for windows

3. Install the required packages using `pip install -r requirements.txt`

4. cd into the algorithm folder using `cd Actor_Critic` or `cd DQN` or `cd PPO`

5. To run Actor Critic, run `python3 test_ac.py`
    To run DQN, run `python3 dqntest.py`
    To run PPO, run `python3 PPO.py`

6. To test the enviorment, run `python3 test_env.py`
<br />

**Congratulations! You are now running our project!**
<br />

NOTE: If you have a CUDA device and would like to use it:
        1. uninstall torch - pip uninstall torch
        2. Go to https://pytorch.org/get-started/locally/ , and select system configurations
        3. Copy the exact command from "Run this command" and run it in the terminal.

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield1]: https://img.shields.io/static/v1?label=Contributor&message=Victor_Li&color=afff75&style=for-the-badge
[contributors-url1]: https://github.com/VictorLi5611
[contributors-shield2]: https://img.shields.io/static/v1?label=Contributor&message=Sebastian_Beimers&color=afff75&style=for-the-badge
[contributors-url2]: https://github.com/sbeimers
[contributors-shield3]: https://img.shields.io/static/v1?label=Contributor&message=Laura_Jin&color=afff75&style=for-the-badge
[contributors-url3]: https://github.com/dxlce
[contributors-shield4]: https://img.shields.io/static/v1?label=Contributor&message=Shawn_Sun&color=afff75&style=for-the-badge
[contributors-url4]: https://github.com/winters21
[contributors-shield5]: https://img.shields.io/static/v1?label=Contributor&message=Evan_Li&color=afff75&style=for-the-badge
[contributors-url5]: https://github.com/Evanli2002