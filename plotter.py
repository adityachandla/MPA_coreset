import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, args, instance_id):
        self.args = args
        self.idx = 0
        self.instance_id = instance_id

    def plot_representative_points(self, representative_points):
        x = [p.coords[0] for p in representative_points]
        y = [p.coords[1] for p in representative_points]
        w = [p.weight for p in representative_points]
        plt.xlim([-10,10])
        plt.ylim([-10,10])
        plt.scatter(x, y, s=w)

    def plot_circle(self, center, radius):
        c = plt.Circle(center, radius, fill=False)
        ax = plt.gca()
        ax.add_patch(c)

    def save_and_clear(self):
        plt.savefig(self.args.output + f"points_plot_{self.instance_id}_{self.idx}.png")
        self.idx += 1
        plt.clf()
