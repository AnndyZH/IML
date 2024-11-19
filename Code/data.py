import json
import pandas as pd
import math
import openpyxl
import os
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from mpl_toolkits.mplot3d import Axes3D
from mplsoccer import Pitch, VerticalPitch, FontManager
from scipy.stats import gaussian_kde

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

sys.stdout.reconfigure(encoding='utf-8')

    
    
def draw_pitch(ax):

    pitch_width = 80    
    pitch_length = 120  
    center_circle_radius = 9.15 
    penalty_area_length = 18  
    penalty_area_width = 44   
    goal_area_length = 6      
    goal_area_width = 20     


    ax.plot([0, 0], [0, pitch_length], color='black') 
    ax.plot([pitch_width, pitch_width], [0, pitch_length], color='black')  
    ax.plot([0, pitch_width], [0, 0], color='black')  
    ax.plot([0, pitch_width], [pitch_length, pitch_length], color='black')  

    ax.plot([0, pitch_width], [pitch_length / 2, pitch_length / 2], color='black')

    center_circle = plt.Circle((pitch_width / 2, pitch_length / 2), center_circle_radius, color='black', fill=False)
    ax.add_artist(center_circle)


    ax.plot([(pitch_width - penalty_area_width) / 2, (pitch_width - penalty_area_width) / 2],
            [0, penalty_area_length], color='black')
    ax.plot([(pitch_width + penalty_area_width) / 2, (pitch_width + penalty_area_width) / 2],
            [0, penalty_area_length], color='black')
    ax.plot([(pitch_width - penalty_area_width) / 2, (pitch_width + penalty_area_width) / 2],
            [penalty_area_length, penalty_area_length], color='black')

    ax.plot([(pitch_width - penalty_area_width) / 2, (pitch_width - penalty_area_width) / 2],
            [pitch_length - penalty_area_length, pitch_length], color='black')
    ax.plot([(pitch_width + penalty_area_width) / 2, (pitch_width + penalty_area_width) / 2],
            [pitch_length - penalty_area_length, pitch_length], color='black')
    ax.plot([(pitch_width - penalty_area_width) / 2, (pitch_width + penalty_area_width) / 2],
            [pitch_length - penalty_area_length, pitch_length - penalty_area_length], color='black')

    ax.plot([(pitch_width - goal_area_width) / 2, (pitch_width - goal_area_width) / 2],
            [0, goal_area_length], color='black')
    ax.plot([(pitch_width + goal_area_width) / 2, (pitch_width + goal_area_width) / 2],
            [0, goal_area_length], color='black')
    ax.plot([(pitch_width - goal_area_width) / 2, (pitch_width + goal_area_width) / 2],
            [goal_area_length, goal_area_length], color='black')

    ax.plot([(pitch_width - goal_area_width) / 2, (pitch_width - goal_area_width) / 2],
            [pitch_length - goal_area_length, pitch_length], color='black')
    ax.plot([(pitch_width + goal_area_width) / 2, (pitch_width + goal_area_width) / 2],
            [pitch_length - goal_area_length, pitch_length], color='black')
    ax.plot([(pitch_width - goal_area_width) / 2, (pitch_width + goal_area_width) / 2],
            [pitch_length - goal_area_length, pitch_length - goal_area_length], color='black')


    penalty_spot_distance = 12 


    ax.scatter(pitch_width / 2, penalty_spot_distance, color='black')


    ax.scatter(pitch_width / 2, pitch_length - penalty_spot_distance, color='black')

 
    ax.set_xlim(0, pitch_width)
    ax.set_ylim(0, pitch_length)
    ax.set_aspect('equal')
    ax.axis('off')

def perceived_length(shot_location):
    left_post_team1 = np.array([0, 36]) 
    right_post_team1 =  np.array([0, 44])
    left_post_team2 = np.array([120, 44]) 
    right_post_team2 = np.array([120, 36])
    first_dist = np.linalg.norm(shot_location - left_post_team1)
    second_dist = np.linalg.norm(shot_location - right_post_team2)
    # print("First distance is " + str(first_dist))
    # print("Second distance is " + str(second_dist))
    if first_dist < second_dist: 
        left_post = left_post_team1
        right_post = right_post_team1
    else:
        left_post = left_post_team2
        right_post = right_post_team2
    # print("Left post loc is " + str(left_post))
    # print("Right post loc is " + str(right_post))
    post_to_post_dist = np.linalg.norm(left_post - right_post) 
    shot_to_left_post = np.linalg.norm(shot_location - left_post)
    shot_to_right_post = np.linalg.norm(shot_location - right_post) 
    # print("Shot loc is " + str(shot_location))
    # print("Post to post "  + str(post_to_post_dist))
    # print("Post to left " + str(shot_to_left_post))
    # print("Post to right " + str(shot_to_right_post))
    cos_theta = (shot_to_left_post ** 2 + shot_to_right_post ** 2 - post_to_post_dist ** 2) / (2 * shot_to_left_post * shot_to_right_post)
    if cos_theta > 1 or cos_theta < -1:
        return 180
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_distance(shot_location):
    x_dist = 120 - shot_location[0]
    y_dist = 0
    if (shot_location[1] < 36):
        y_dist = 36 - shot_location[1]
    elif (shot_location[1] > 44):
        y_dist = shot_location[1] - 44
    return math.sqrt(x_dist ** 2 + y_dist ** 2)


def save_dataframe(folder_path, output_path):
    json_files = glob.glob(f'{folder_path}/*.json')
    shot_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for match in data:
                if match['type']['name'] == "Shot" and match['shot']['type']['name'] != "Penalty":
                    
                    goalkeeper_location = None

                    freeze_frame = match['shot'].get('freeze_frame', None)
                    if freeze_frame:
                        for frame in freeze_frame:
                            if not frame.get('teammate', False) and frame.get('position', {}).get('name') == 'Goalkeeper':
                                goalkeeper_location = frame.get('location', None)
                                break
                                
                                
                    shooter_location = match['location']
                    distance = None
                    if shooter_location is not None and goalkeeper_location is not None:
                        distance = np.sqrt((goalkeeper_location[0] - shooter_location[0])**2 + (goalkeeper_location[1] - shooter_location[1])**2)        
                    shot_data.append({
                        'Time': match['timestamp'],
                        'Team': match['team']['name'],
                        'Player': match['player']['name'],
                        'Location': match['location'],
                        'GK': goalkeeper_location,
                        'GK_dist': distance,
                        'Outcome': match['shot']['outcome']['name'],
                        'Technique': match['shot']['technique']['name'],
                        'Body_part': match['shot']['body_part']['name'],
                        'xG': match['shot']['statsbomb_xg'],
                        'freeze_frame': match['shot'].get('freeze_frame', None)
                    })
    
    df = pd.DataFrame(shot_data)
    df.to_pickle(output_path + 'data.pkl')

def gk_dist_to_goal(goalkeeper_location):
    return math.sqrt((goalkeeper_location[0] - 120)**2 + (goalkeeper_location[1] - 40)**2)      

def dist_to_gk(df):
    shooter_location = df['Location']
    goalkeeper_location = df['GK']
    
    if shooter_location is not None and goalkeeper_location is not None:
        distance = np.sqrt((goalkeeper_location[0] - shooter_location[0])**2 + 
                           (goalkeeper_location[1] - shooter_location[1])**2)
        return distance
    else:
        return np.nan
        
        
def gk_dist_to_goal_y(goalkeeper_location):
    return goalkeeper_location[1] - 40

#Barycentric Coordinate Method
def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def is_point_in_triangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)
     
def player_in_triangle(location, freeze_frame):
    goalpost_left = [120, 36]    
    goalpost_right = [120, 44]
    count = 0
    for players in freeze_frame:
        if is_point_in_triangle(players['location'], location, goalpost_left, goalpost_right):
            if players['teammate'] == False:
                count = count + 1.2
            else:
                count = count + 0.8
    return count


def is_header(Body_Part):
    if Body_Part == 'Head':
        return 1
    return 0
     
def three_meters_away(location, freeze_frame):
    num = 0
    for players in freeze_frame:
        player_loc = players['location']
        if math.sqrt((location[0] - player_loc[0]) ** 2 + (location[1] - player_loc[1]) ** 2) <= 3:
            if players['teammate'] == False:
                num = num + 1.2
            else:
                num = num + 0.8
    return num     
#not using
def save_data(team, folder_path, output_path):
    #'D:/SoccerMatchData/data/events'
    shot = []
    goal_bool = np.array([])
    goal = []
    non_goal = []
    xG_goal = []
    xG_non_goal = []
    json_files = glob.glob(f'{folder_path}/*.json')

    for json_file in json_files:
        flag = True
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # lineup_data = []
            if (data[0]['type']['name'] == 'Starting XI' and data[0]['team']['name'] == team) or (data[1]['type']['name'] == 'Starting XI' and data[1]['team']['name'] == team):
                flag = False
        
        game_data = []
        
        i = 0
        if flag:
            continue
            
        for match in data:
            # if 'tactics' in match and 'lineup' in match['tactics']:
            #     for player in match['tactics']['lineup']:
            #         lineup_data.append({
            #             'Team': match['team']['name'],
            #             'Player Name': player['player']['name'],
            #             'Position': player['position']['name'],
            #             'Jersey Number': player['jersey_number']
            #         })
            if match['type']['name'] == "Shot" and match['team']['name'] == team:
                game_data.append({
                    'Time': match['timestamp'],
                    'Team': match['team']['name'],
                    'Player': match['player']['name'],
                    'Location': match['location'],
                    'Outcome': match['shot']['outcome']['name'],
                    'Technique': match['shot']['technique']['name'],
                    'Body_part': match['shot']['body_part']['name'],
                    'xG': match['shot']['statsbomb_xg']
                })

        df = pd.DataFrame(game_data)
        df_goals = (df[df['Outcome'] == 'Goal'])
        df_non_goal = (df[df['Outcome'] != 'Goal'])
        
        for index, row in df.iterrows():
            shot.append(row['Location'])
            if row['Outcome'] == 'Goal':
                goal_bool = np.append(goal_bool, 1)
            else:
                goal_bool = np.append(goal_bool, 0)
                
        for index, row in df_goals.iterrows():
            location = row['Location']
            goal.append(location)
            xg = row['xG']
            xG_goal.append(xg)
            
        for index, row in df_non_goal.iterrows():
            location = row['Location']
            non_goal.append(location)
            xg = row['xG']
            xG_non_goal.append(xg)
    
    goal_coor = np.array(goal)
    non_goal_coor = np.array(non_goal)
    xG_g = np.array(xG_goal)
    xG_non_g = np.array(xG_non_goal)
    shoot = np.array(shot)
    np.save(output_path+'/goal_coor.npy', goal_coor)
    np.save(output_path+'/non_goal_coor.npy', non_goal_coor)
    np.save(output_path+'/xG_g.npy', xG_g)
    np.save(output_path+'/xG_non_g.npy', xG_non_g) 
    np.save(output_path+'/goal_bool.npy', goal_bool) 
    np.save(output_path+'/shot_location.npy', shoot)      
#not using
def extract_data(output_path):
    return np.load(output_path)

def heatmap_xG_sum(x, y, xG):
    pitch = VerticalPitch(
        pad_bottom=0.5,  # pitch extends slightly below halfway line
        half=True,  # half of a pitch
        goal_type='box',
        goal_alpha=0.8, 
        pitch_color='#22312b', 
        line_color='#c7d5cc'
    )  

    fig, ax = pitch.draw(figsize=(12, 10))
    bin_statistic = pitch.bin_statistic(
        x, y, statistic = 'sum', bins=(36,30), values=xG
    )
    pcm = pitch.heatmap(
        bin_statistic, ax=ax, cmap='viridis', edgecolors='grey'
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Total xG')
    plt.show()
    
def heatmap_xG_mean(x, y, xG):
    pitch = VerticalPitch(
        pad_bottom=0.5,  # pitch extends slightly below halfway line
        half=True,  # half of a pitch
        goal_type='box',
        goal_alpha=0.8, 
        pitch_color='#22312b', 
        line_color='#c7d5cc'
    )  

    fig, ax = pitch.draw(figsize=(12, 10))
    bin_statistic = pitch.bin_statistic(
        x, y, statistic = 'mean', bins=(36,30), values=xG
    )
    pcm = pitch.heatmap(
        bin_statistic, ax=ax, cmap='viridis', edgecolors='grey'
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Total xG')
    plt.show()

def heatmap_shot(x, y):
    pitch = VerticalPitch(
        pad_bottom=0.5,  # pitch extends slightly below halfway line
        half=True,  # half of a pitch
        goal_type='box',
        goal_alpha=0.8, 
        pitch_color='#22312b', 
        line_color='#c7d5cc'
    )  

    fig, ax = pitch.draw(figsize=(12, 10))
    bin_statistic = pitch.bin_statistic(
        x, y, statistic = 'count', bins=(48,40)
    )
    pcm = pitch.heatmap(
        bin_statistic, ax=ax, cmap='viridis', edgecolors='grey'
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('shot')
    plt.show()

    
def logistic_regression(pLength, dist, gk_dist,gk_goal, gk_goal_y, three_away, in_triangle, Body_part, goal):
    model_name = 'Logistic'
    models = {}
    models['Logistic'] = {}
    
    #X = np.column_stack((pLength, angle))
    X = np.column_stack((pLength, dist, gk_dist, gk_goal, gk_goal_y, three_away, in_triangle, Body_part))
    y = goal
    
    models['Logistic']['model'] = LogisticRegression()
    models[model_name]['model'].fit(X, y)
    models[model_name]['y_pred'] = models[model_name]['model'].predict_proba(X)[:,1]
    coefficients = models[model_name]['model'].coef_
    intercept = models[model_name]['model'].intercept_
    models[model_name]['coefficients'] = coefficients
    models[model_name]['intercept'] = intercept
    
    return models

def perceivedLength3d(shot_location):
    global left_post_team1, right_post_team2, left_post_team2, right_post_team1, mid_top_team1, mid_top_team2, mid_ground_team1, mid_ground_team2
    first_dist = np.linalg.norm(shot_location - left_post_team1)
    second_dist = np.linalg.norm(shot_location - right_post_team2)
    # print("First distance is " + str(first_dist))
    # print("Second distance is " + str(second_dist))
    if first_dist < second_dist: 
        left_post = left_post_team1
        right_post = right_post_team1
        mid_top = mid_top_team1
        mid_ground = mid_ground_team1
    else:  
        left_post = left_post_team2
        right_post = right_post_team2
        mid_top = mid_top_team2
        mid_ground = mid_ground_team2

    #2d 
    post_to_post_dist = np.linalg.norm(left_post - right_post) 
    shot_to_left_post = np.linalg.norm(shot_location - left_post)  
    shot_to_right_post = np.linalg.norm(shot_location - right_post) 

    top_to_shot = np.linalg.norm(shot_location - mid_top) 
    ground_mid_to_shot = np.linalg.norm(shot_location - mid_ground) 
    top_to_mid = np.linalg.norm(mid_top - mid_ground) 

    cos_theta_1d = (shot_to_left_post**2 + shot_to_right_post**2 - post_to_post_dist**2) / (2 * shot_to_left_post * shot_to_right_post)
    cos_theta_2d = cos_theta_2d = (top_to_shot**2 + ground_mid_to_shot**2 - top_to_mid**2) / (2 * top_to_shot * ground_mid_to_shot)

    angle_rad = np.arccos(cos_theta_1d)  
    angle_rad_2 = np.arccos(cos_theta_2d)

    angle_deg = np.degrees(angle_rad)
    angle_deg_2 = np.degrees(angle_rad_2)
    pLength = angle_deg * angle_deg_2
    return pLength


def plot_logistic_regression(pLength, dist, models):
    plt.scatter(pLength, models['Logistic']['y_pred'], label='Predicted xG', color='blue', alpha=0.6)
    coefficients = models['Logistic']['coefficients'][0]  
    intercept = models['Logistic']['intercept'][0] 
    
    pLength_range = np.linspace(min(pLength), max(pLength), 100)
    distance_range = np.mean(dist)
    logistic_curve = 1 / (1 + np.exp(-(intercept + coefficients[0] * pLength_range + coefficients[1] * distance_range)))
    
    plt.plot(pLength_range, logistic_curve, label='Logistic Regression', color='red')
    plt.xlabel('pLength')
    plt.ylabel('Predicted xG')
    plt.title('Logistic Regression: Predicted xG vs pLength')
    plt.grid(True)
    plt.legend()
    plt.show()

#not using    
def plot_logistic_regression_3D(pLength, distance, models):
    coefficients = models['Logistic']['coefficients'][0] 
    intercept = models['Logistic']['intercept'][0] 

    pLength_range = np.linspace(min(pLength), max(pLength), 100)
    distance_range = np.linspace(min(distance), max(distance), 100)
    pLength_grid, distance_grid = np.meshgrid(pLength_range, distance_range)

    logistic_curve = 1 / (1 + np.exp(-(intercept 
                                       + coefficients[0] * pLength_grid 
                                       + coefficients[1] * distance_grid)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(pLength_grid, distance_grid, logistic_curve, cmap='viridis', alpha=0.6)

    ax.scatter(pLength, distance, models['Logistic']['y_pred'], color='blue', label='Predicted xG', alpha=0.6)

    ax.set_xlabel('pLength')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Predicted xG')
    ax.set_title('Logistic Regression: pLength and Distance vs Predicted xG')

    plt.legend()
    plt.show()
#not using  
def plot_logistic_regression_contour(pLength, distance, models):
    coefficients = models['Logistic']['coefficients'][0]
    intercept = models['Logistic']['intercept'][0]

    pLength_range = np.linspace(min(pLength), max(pLength), 100)
    distance_range = np.linspace(min(distance), max(distance), 100)
    pLength_grid, distance_grid = np.meshgrid(pLength_range, distance_range)
    logistic_curve = 1 / (1 + np.exp(-(intercept 
                                       + coefficients[0] * pLength_grid 
                                       + coefficients[1] * distance_grid)))

    plt.contourf(pLength_grid, distance_grid, logistic_curve, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Predicted xG')

    plt.scatter(pLength, distance, c=models['Logistic']['y_pred'], edgecolor='black', label='Predicted xG', alpha=0.6)

    plt.xlabel('pLength')
    plt.ylabel('Distance')
    plt.title('Logistic Regression: pLength and Distance vs Predicted xG (Contour)')
    plt.legend()
    plt.show()    
#not using        
def plot_distance_vs_xg(pLength, distance, models):
    coefficients = models['Logistic']['coefficients'][0] 
    intercept = models['Logistic']['intercept'][0]  
    
    pLength_fixed = np.mean(pLength) 
    
    distance_range = np.linspace(min(distance), max(distance), 100)
    
    logistic_curve = 1 / (1 + np.exp(-(intercept 
                                       + coefficients[0] * pLength_fixed 
                                       + coefficients[1] * distance_range)))  
    
    plt.plot(distance_range, logistic_curve, label='Logistic Regression (distance effect)', color='red')
    
    plt.scatter(distance, models['Logistic']['y_pred'], color='blue', alpha=0.6, label='Predicted xG')
    
    plt.xlabel('Distance')
    plt.ylabel('Predicted xG')
    plt.title('Logistic Regression: Distance vs Predicted xG')
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_r2(y_true, y_pred):
    
    return metrics.r2_score(y_true, y_pred)

def logistic_regression_formula(coef, intercept, feature_names):
    terms = [f"{c}*{name}" for c, name in zip(coef, feature_names)]
    formula = f"1 / (1 + exp(-({intercept} + {' + '.join(terms)})))"
    return formula


if __name__ == "__main__":
    #save_dataframe('D:/SoccerMatchData/data/events', 'C:/Users/Administrator/IML/')
    df = pd.read_pickle('C:/Users/Administrator/IML/data.pkl')

    gk_dist = []
    pLength = []
    dist = []
    goal_bool = []
    gk_goal = []
    gk_goal_y = []
    three_away = []
    xg = []
    in_triangle = []
    Body_part = []
    for index, rows in df.iterrows():
        if np.isnan(rows['GK_dist']):
            continue
        gk_dist.append(rows['GK_dist'])
        gk_goal.append(gk_dist_to_goal(rows["GK"]))
        three_away.append(three_meters_away(rows['Location'], rows['freeze_frame']))
        gk_goal_y.append(gk_dist_to_goal_y(rows["GK"]))
        in_triangle.append(player_in_triangle(rows['Location'], rows['freeze_frame']))
        Body_part.append(is_header(rows['Body_part']))
        pLength.append(perceived_length(rows['Location']))
        dist.append(calculate_distance(rows['Location']))
        xg.append(rows['xG'])
        if rows['Outcome'] == 'Goal':
            goal_bool.append(1)
        else:
            goal_bool.append(0)   
    model = logistic_regression(pLength, dist, gk_dist, gk_goal, gk_goal_y, three_away, in_triangle, Body_part, goal_bool)
    intercept = model['Logistic']['intercept']
    coefficients = model['Logistic']['coefficients']  
    feature_names = ["pLength", "dist", "gk_dist"]
    print(logistic_regression_formula(coefficients, intercept, feature_names))
    
    
    # coefficients = model['Logistic']['coefficients']
    # intercept = model['Logistic']['intercept']
    y_pred = model['Logistic']['y_pred']

    y_true = goal_bool
    r2_statsbomb = calculate_r2(y_true, xg)
    r2_value = calculate_r2(y_true, y_pred)
    print(r2_value, r2_statsbomb)




