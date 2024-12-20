from utils import read_video, save_video
from suivi import Tracker
import cv2
import numpy as np
from teams import AssignerEquipe
from balltoplayer import Balltoplayer
from camera import EstimerMouvementdecamera
from view_transformer import ViewTransformer
from distancespeed import CalculSpeedwdistance


def main():
    # Paths to the icons (update with actual paths)
    speed_icon_path = "icons/speedometer (1).png"  # Replace with the actual path to the speed icon
    distance_icon_path = "icons/rotate (1).png"  # Replace with the actual path to the distance icon

    videoparts = read_video('input/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.getsuivijoueur(videoparts,
                                    read_from_stub=False,
                                    stub_path='stubs/track_stubs.pkl')
    # Get object positions
    tracker.ajouterposition(tracks)

    # Camera movement estimator
    mouvementcam = EstimerMouvementdecamera(videoparts[0])
    mouvementcameraparpart = mouvementcam.mouvementdecam(videoparts,
                                                                         read_from_stub=False,
                                                                         stub_path='stubs/camera_movement_stub.pkl')
    mouvementcam.ajouterpositions(tracks, mouvementcameraparpart)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.positionduballon(tracks["ball"])

    # Speed and distance estimator
    speedwdistance = CalculSpeedwdistance()
    speedwdistance.ajoutspeeddist(tracks)

    # Assign Player Teams
    assignerequipe = AssignerEquipe()
    assignerequipe.couleurdequipe(videoparts[0],
                                 tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            equipe = assignerequipe.equipedejoueur(videoparts[frame_num],
                                                track['bbox'],
                                                player_id)
            tracks['players'][frame_num][player_id]['team'] = equipe
            tracks['players'][frame_num][player_id]['team_color'] = assignerequipe.couleur_equ[equipe]





    #possession
    assignerjoueur = Balltoplayer()
    possession = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = assignerjoueur.assignerball(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            possession.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            possession.append(possession[-1])
    possession = np.array(possession)

    # Draw output
    outputvidframe = tracker.draw_annotations(videoparts, tracks, possession)

    outputvidframe = mouvementcam.orsemcammov(outputvidframe, mouvementcameraparpart)

    speedwdistance.orsemspeeddist(outputvidframe, tracks, speed_icon_path, distance_icon_path)

    # Save video
    save_video(outputvidframe, 'output/bvb.avi')


if __name__ == '__main__':
    main()
