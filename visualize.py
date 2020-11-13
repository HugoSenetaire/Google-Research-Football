import matplotlib.patches as patches
from  matplotlib.patches import Arc
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches
from enum import Enum
import math
import numpy as np
from IPython.display import HTML

WIDTH = 105
HEIGHT = 68
X_RESIZE = WIDTH
Y_RESIZE = HEIGHT / 0.42



class GameMode(Enum):
  Normal = 0
  KickOff = 1
  GoalKick = 2
  FreeKick = 3
  Corner = 4
  ThrowIn = 5
  Penalty = 6

def scale_x(x):
  return (x + 1) * (X_RESIZE/2)

def scale_y(y):
  return (y + 0.42) * (Y_RESIZE/2)

def draw_team(obs, team, side):
    X = []
    Y = []
    for x, y in obs[side]:
      X.append(x)
      Y.append(y)
    team.set_data(X, Y)

def draw_ball(obs, ball):
  ball.set_markersize(10 + 5 * obs["ball"][2]) # Scale size of ball based on height
  ball.set_data(obs["ball"][:2])

def draw_ball_owner(obs, ball_owner, team_active):
  if obs["ball_owned_team"] == 0:
    x, y = obs["left_team"][obs["ball_owned_player"]]
    ball_owner.set_data(x, y)
    team_active.set_data(WIDTH / 4 + 7, -7)
    team_active.set_markerfacecolor("red")
  elif obs["ball_owned_team"] == 1:
    x, y = obs["right_team"][obs["ball_owned_player"]]
    ball_owner.set_data(x, y)
    team_active.set_data(WIDTH / 4 + 50, -7)
    team_active.set_markerfacecolor("blue")
  else:
    ball_owner.set_data([], [])
    team_active.set_data([], [])
    
def draw_players_directions(obs, directions, side):
  index = 0
  if "right" in side:
    index = 11
  for i, player_dir in enumerate(obs[f"{side}_direction"]):
    x_dir, y_dir = player_dir
    dist = math.sqrt(x_dir ** 2 + y_dir ** 2) + 0.00001 # to prevent division by 0
    x = obs[side][i][0]
    y = obs[side][i][1] 
    directions[i + index].set_data([x, x + x_dir / dist ], [y, y + y_dir / dist])

def extract_data(frame):
  res = {}
  obs = frame[0]['observation']['players_raw'][0]
  res["left_team"] = [(scale_x(x), scale_y(y)) for x, y in obs["left_team"]]
  res["right_team"] = [(scale_x(x), scale_y(y)) for x, y in obs["right_team"]]

  ball_x, ball_y, ball_z = obs["ball"]
  res["ball"] = [scale_x(ball_x),  scale_y(ball_y), ball_z]
  res["score"] = obs["score"]
  res["steps_left"] = obs["steps_left"]
  res["ball_owned_team"] = obs["ball_owned_team"]
  res["ball_owned_player"] = obs["ball_owned_player"]
  res["right_team_roles"] = obs["right_team_roles"]
  res["left_team_roles"] = obs["left_team_roles"]
  res["left_team_direction"] = obs["left_team_direction"]
  res["right_team_direction"] = obs["right_team_direction"]
  res["game_mode"] = GameMode(obs["game_mode"]).name
  return res

def drawPitch(width, height, color="w"):

    fig = plt.figure()
    ax = plt.axes(xlim=(-10, width + 10), ylim=(-15, height + 5))
    plt.axis('off')

    # Grass around pitch
    rect = patches.Rectangle((-5,-5), width + 10, height + 10, linewidth=1, edgecolor='gray',facecolor='#3f995b', capstyle='round')
    ax.add_patch(rect)

    # Pitch boundaries
    rect = plt.Rectangle((0, 0), width, height, ec=color, fc="None", lw=2)
    ax.add_patch(rect)

    # Middle line
    plt.plot([width/2, width/2], [0, height], color=color, linewidth=2)
    
    # Dots
    dots_x = [11, width/2, width-11]
    for x in dots_x:
      plt.plot(x, height/2, 'o', color=color, linewidth=2)

    # Penalty box  
    penalty_box_dim = [16.5, 40.3]
    penalty_box_pos_y = (height - penalty_box_dim[1]) / 2

    rect = plt.Rectangle((0, penalty_box_pos_y), penalty_box_dim[0], penalty_box_dim[1], ec=color, fc="None", lw=2)
    ax.add_patch(rect)
    rect = plt.Rectangle((width, penalty_box_pos_y), -penalty_box_dim[0], penalty_box_dim[1], ec=color, fc="None", lw=2)
    ax.add_patch(rect)

    #Goal box
    goal_box_dim = [5.5, penalty_box_dim[1] - 11 * 2]
    goal_box_pos_y = (penalty_box_pos_y + 11)

    rect = plt.Rectangle((0, goal_box_pos_y), goal_box_dim[0], goal_box_dim[1], ec=color, fc="None", lw=2)
    ax.add_patch(rect)
    rect = plt.Rectangle((width, goal_box_pos_y), -goal_box_dim[0], goal_box_dim[1], ec=color, fc="None", lw=2)
    ax.add_patch(rect)

    #Goals
    rect = plt.Rectangle((0, penalty_box_pos_y + 16.5), -3, 7.5, ec=color, fc=color, lw=2, alpha=0.3)
    ax.add_patch(rect)
    rect = plt.Rectangle((width, penalty_box_pos_y + 16.5), 3, 7.5, ec=color, fc=color, lw=2, alpha=0.3)
    ax.add_patch(rect)
      
    # Middle circle
    mid_circle = plt.Circle([width/2, height/2], 9.15, color=color, fc="None", lw=2)
    ax.add_artist(mid_circle)


    # Penalty box arcs
    left  = patches.Arc([11, height/2], 2*9.15, 2*9.15, color=color, fc="None", lw=2, angle=0, theta1=308, theta2=52)
    ax.add_patch(left)
    right = patches.Arc([width - 11, height/2], 2*9.15, 2*9.15, color=color, fc="None", lw=2, angle=180, theta1=308, theta2=52)
    ax.add_patch(right)

    # Arcs on corners
    corners = [[0, 0], [width, 0], [width, height], [0, height]]
    angle = 0
    for x,y in corners:
      c = patches.Arc([x, y], 2, 2, color=color, fc="None", lw=2, angle=angle,theta1=0, theta2=90)
      ax.add_patch(c)
      angle += 90
    return fig, ax

# Change size of figure
plt.rcParams['figure.figsize'] = [20, 16]
fig, ax = drawPitch(WIDTH, HEIGHT)
ax.invert_yaxis()

class Visualizer():

  def __init__(self,output, steps=100):
    self.output = output
    self.steps= steps

    self.ball_owner, = ax.plot([], [], 'o', markersize=30,  markerfacecolor="yellow", alpha=0.5)
    self.team_active, = ax.plot([], [], 'o', markersize=30,  markerfacecolor="blue", markeredgecolor="None")

    self.team_left, = ax.plot([], [], 'o', markersize=20, markerfacecolor="r", markeredgewidth=2, markeredgecolor="white")
    self.team_right, = ax.plot([], [], 'o', markersize=20,  markerfacecolor="b", markeredgewidth=2, markeredgecolor="white")

    self.ball, = ax.plot([], [], 'o', markersize=10,  markerfacecolor="black", markeredgewidth=2, markeredgecolor="white")
    self.text_frame = ax.text(-5, -5, '', fontsize=25)
    self.match_info = ax.text(105 / 4 + 10, -5, '', fontsize=25)
    self.game_mode = ax.text(105 - 25, -5, '', fontsize=25)
    self.goal_notification = ax.text(105 / 4 + 10, 0, '', fontsize=25)

    # Drawing of directions definitely can be done in a better way
    self.directions = []
    for i in range(22):
      direction, = ax.plot([], [], color='yellow', lw=3)
      self.directions.append(direction)

    self.drawings = [self.team_active, self.ball_owner, self.team_left, self.team_right, self.ball, self.text_frame, self.match_info, self.game_mode, self.goal_notification]


  def init(self):
      self.team_left.set_data([], [])
      self.team_right.set_data([], [])
      self.ball_owner.set_data([], [])
      self.team_active.set_data([], [])
      self.ball.set_data([], [])
      return self.drawings 

  def animate(self,i):
    global prev_score_a, prev_score_b
    obs = extract_data(self.output[i])

    # Draw info about ball possesion
    draw_ball_owner(obs, self.ball_owner, self.team_active)

    # Draw players
    draw_team(obs, self.team_left, "left_team")
    draw_team(obs, self.team_right, "right_team")

    draw_players_directions(obs, self.directions, "left_team")
    draw_players_directions(obs, self.directions, "right_team")
      
    draw_ball(obs, self.ball)

    # Draw textual informations
    self.text_frame.set_text(f"Frame: {i}/{obs['steps_left'] + i - 1}")
    self.game_mode.set_text(f"Game mode: {obs['game_mode']}")
    
    self.score_a, self.score_b = obs["score"]
    self.match_info.set_text(f"Left team {self.score_a} : {self.score_b} Right Team")

    return self.drawings

  def get_animation(self):
    # May take a while
    anim = animation.FuncAnimation(fig, self.animate, init_func=self.init,
                                frames=self.steps, interval=100, blit=True)
    
    return anim



  #HTML(anim.to_html5_video())
