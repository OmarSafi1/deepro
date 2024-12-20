import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class Balltoplayer():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assignerball(self, players, ball_bbox):
        positiondeballe = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        joueurassigne=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),positiondeballe)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),positiondeballe)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    joueurassigne = player_id

        return joueurassigne