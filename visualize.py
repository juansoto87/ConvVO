import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--pose_dir", type=str, required=True, help="where the the ground truth is stored")
parser.add_argument("--model_folder", type=str, required=True, help="where the predicted poses file is stored")
parser.add_argument("--seq", type=str, required=True, help="which sequence is going to be evaluated")
parser.add_argument("--model_name", type=str, required=True, help="name of model")
parser.add_argument("--type_calc", type=str, required=True, help="tras, rot or total")
args = parser.parse_args()



pose_GT_dir = args.pose_dir
predicted_result_dir = os.path.join(args.model_folder, ('poses_'+ args.model_name +'_'+ args.seq + '.npy'))
gradient_color = True

def plot_route(gt, out, c_gt='g', c_out='r'):
	x_idx = 3
	y_idx = 5
	x = [v for v in gt[:, x_idx]]
	y = [v for v in gt[:, y_idx]]
	plt.plot(x, y, color=c_gt, label='Ground Truth')
	#plt.scatter(x, y, color='b')

	x = [v for v in out[:, x_idx]]
	y = [v for v in out[:, y_idx]]
	plt.plot(x, y, color=c_out, label='DeepVO')
	#plt.scatter(x, y, color='b')
	plt.gca().set_aspect('equal', adjustable='datalim')



GT_pose_path = os.path.join(pose_GT_dir,(args.seq + '.npy'))
gt = np.load(GT_pose_path)
if args.type_calc == 'rot':
	out = np.load(predicted_result_dir)
	out = np.append(out, np.zeros((out.shape[0], 3)), axis = 1)
elif args.type_calc == 'tras':
	out = np.load(predicted_result_dir)
	out = np.append(np.zeros((out.shape[0], 3)), out, axis = 1)
# print(out[0:5,])
# print(out.shape)
mse_rotate = 100 * np.mean((out[:, :3] - gt[:, :3])**2)
mse_translate = np.mean((out[:, 3:] - gt[:, 3:6])**2)
print('mse_rotate: ', mse_rotate)
print('mse_translate: ', mse_translate)


if gradient_color:
	# plot gradient color
	step = 200
	plt.clf()
	plt.scatter([gt[0][3]], [gt[0][5]], label='sequence start', marker='s', color='k')
	for st in range(0, len(out), step):
		end = st + step
		g = max(0.2, st/len(out))
		c_gt = (0, g, 0)
		c_out = (1, g, 0)
		plot_route(gt[st:end], out[st:end], c_gt, c_out)
		if st == 0:
			plt.legend()
		plt.title('Sequence' + args.seq)
		save_name = '{}route_{}_gradient.png'.format((args.model_name), args.seq)
	plt.savefig(os.path.join(args.model_folder,save_name))
else:
	# plot one color
	plt.clf()
	plt.scatter([gt[0][3]], [gt[0][5]], label='sequence start', marker='s', color='k')
	plot_route(gt, out, 'r', 'b')
	plt.legend()
	plt.title('Sequence' + args.seq)
	save_name = '{}route_{}.png'.format(args.model_name, args.seq)
	plt.savefig(os.path.join(args.model_folder,save_name))
