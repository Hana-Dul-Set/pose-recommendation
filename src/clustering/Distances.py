import math
#from clustering.AnglePreprocessor import angle_triplets
angle_triplets = [(3, 1, 0), (4, 2, 0), (6, 5, 0), (7, 5, 0), (8, 6, 5), (9, 7, 5), (10, 8, 6), (11, 9, 7), (12, 14, 5),
                   (13, 14, 5), (14, 5, 0), (15, 12, 14), (16, 13, 14), (17, 15, 12), (18, 16, 13), (19, 21, 17), (20, 19, 21), 
                   (21, 17, 15), (22, 24, 18), (23, 22, 24), (24, 18, 16)]

point_groups = [[0, 1, 2, 3, 4], [19, 20, 21], [22, 23, 24]]  #face, lfoot, rfoot

#perform
def point_distance(a, b):
   return math.sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))]))

def distance(A, B):
   total_dist = 0

   total_dist += point_distance(A[0:3], B[0:3]) * 5

   A_pose, B_pose = A[3:], B[3:]
   for i in range(len(A_pose)):
      dist = point_distance(A_pose[i*3:(i+1)*3], B_pose[i*3:(i+1)*3])
      for group in point_groups:
         if i in group:
            dist *= 1 / len(group)
      total_dist += dist
   return total_dist

angle_point_groups = [[0, 1, 2,], [19, 21], [22, 24]]  #face, lfoot, rfoot

def angle_distance(A, B):
   total_dist = 0
   #nose pose diff
   total_dist += point_distance(A[0:2], B[0:2]) * 5
   
   #angle diff
   poseA, poseB = A[2:], B[2:]
   for i in range(0, len(poseA), 2):
      dist = point_distance(poseA[i:i+1], poseB[i:i+1])
      
      for group in angle_point_groups:
         if angle_triplets[i//2][1] in group:
            dist *= 1 / len(group)
      total_dist += dist
   return total_dist