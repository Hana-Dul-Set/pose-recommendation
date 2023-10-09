import math

#perform
def point_distance(a, b):
   return math.sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))]))

def distance(A, B):
   total_dist = 0

   total_dist += point_distance(A[0:3], B[0:3]) * 5

   A_pose, B_pose = A[3:], B[3:]
   point_groups = [[0, 1, 2, 3, 4], [19, 20, 21], [22, 23, 24]]  #face, lfoot, rfoot
   for i in range(len(A_pose)):
      dist = point_distance(A_pose[i*3:(i+1)*3], B_pose[i*3:(i+1)*3])
      for group in point_groups:
         if i in group:
            dist *= 1 / len(group)
      total_dist += dist
   return total_dist