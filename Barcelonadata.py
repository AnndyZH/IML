import json
import pandas as pd
import openpyxl
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from mplsoccer import Pitch, VerticalPitch, FontManager
from scipy.stats import gaussian_kde

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


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
        print(shot_location)
        return 180
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

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
                #print(match['shot']['statsbomb_xg'])

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
    print(goal)
    np.save(output_path+'/goal_coor.npy', goal_coor)
    np.save(output_path+'/non_goal_coor.npy', non_goal_coor)
    np.save(output_path+'/xG_g.npy', xG_g)
    np.save(output_path+'/xG_non_g.npy', xG_non_g) 
    np.save(output_path+'/goal_bool.npy', goal_bool) 
    np.save(output_path+'/shot_location.npy', shoot)      

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

def logistic_regression(pLength, angle, goal):
    model_name = 'Logistic'
    models = {}
    models['Logistic'] = {}
    
    #X = np.column_stack((pLength, angle))
    X = np.array(pLength).reshape(-1, 1)
    y = goal
    
    models['Logistic']['model'] = LogisticRegression()
    models[model_name]['model'].fit(X, y)
    models[model_name]['y_pred'] = models[model_name]['model'].predict_proba(X)[:,1]
    return models   
    
def plot_logistic_regression(pLength, models):
    plt.scatter(pLength, models['Logistic']['y_pred'], label='Predicted xG', color='blue', alpha=0.6)
    plt.xlabel('pLength')
    plt.ylabel('Predicted xG')
    plt.title('Logistic Regression: Predicted xG vs pLength')
    plt.grid(True)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
        
    #save_data('Barcelona', 'D:/SoccerMatchData/data/events', 'D:/SoccerMatchData/data/try')
    xG = extract_data('D:/SoccerMatchData/data/try/xG_g.npy')
    goal_coor = extract_data('D:/SoccerMatchData/data/try/goal_coor.npy')
    non_goal_coor = extract_data('D:/SoccerMatchData/data/try/non_goal_coor.npy')
    shot_location = extract_data('D:/SoccerMatchData/data/try/shot_location.npy')
    pLength = extract_data('D:/SoccerMatchData/data/try/perceived_length.npy')
    goal_bool = extract_data('D:/SoccerMatchData/data/try/goal_bool.npy')
    print(len(pLength))
    # pLength = np.array([])
    # for i in shot_location:
    #     pLength = np.append(pLength, perceived_length(i))
    # np.save('D:/SoccerMatchData/data/try' + '/perceived_length.npy', pLength)
    #np.set_printoptions(threshold=np.inf)
    #print(pLength)
    angle = []
    model = logistic_regression(pLength, angle, goal_bool)
    plot_logistic_regression(pLength, model)
    #print(goal_coor)
    # x = goal_coor[:, 0]
    # y = goal_coor[:, 1]
    #draw_pitch(ax)


    # pitch = VerticalPitch(pad_bottom=0.5,  # pitch extends slightly below halfway line
    #                       half=True,  # half of a pitch
    #                       goal_type='box',
    #                       goal_alpha=0.8, 
    #                       pitch_color='#22312b', 
    #                       line_color='#c7d5cc')  # control the goal transparency

    # fig, ax = pitch.draw(figsize=(12, 10))
    # ax.scatter(x, y, c='#ad993c', s = 5, label='Goal')




    # #ax.scatter(non_goal_coor[:, 1], non_goal_coor[:, 0], c='#ba4f45', s = 5, label='No goal')

    # #ax.legend()
    # plt.show()
    #df = pd.DataFrame(lineup_data)

    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.max_columns', None)  # Show all columns
    # with open('C:/Users/Administrator/Desktop/15946.json', 'r') as f:
    #     data = json.load(f)

        # excel_file_name = os.path.splitext(os.path.basename(json_file))[0] + '.xlsx'
        # excel_file_path = os.path.join(output_path, excel_file_name)
        # print(excel_file_path)
        # df.to_excel(excel_file_path, index=False)


    # workbook = openpyxl.load_workbook('C:/Users/Administrator/Desktop/test.xlsx')
    # worksheet = workbook['Sheet1']
    # worksheet.delete_rows(1, worksheet.max_row)
    # df.to_excel('C:/Users/Administrator/Desktop/test.xlsx', index=False)
    # print(df)


