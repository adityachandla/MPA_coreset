import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, args, instance_id):
        self.args = args
        self.idx = 0
        self.instance_id = instance_id
        self.limit = args.center_limit

    def plot_representative_points(self, representative_points):
        x = [p.coords[0] for p in representative_points]
        y = [p.coords[1] for p in representative_points]
        w = [p.weight for p in representative_points]
        plt.xlim(self.limit)
        plt.ylim(self.limit)
        plt.scatter(x, y, s=w)

    def plot_circle(self, center, radius):
        c = plt.Circle(center, radius, fill=False)
        ax = plt.gca()
        ax.add_patch(c)

    def save_and_clear(self):
        plt.savefig(self.args.output + f"points_plot_{self.instance_id}_{self.idx}.png")
        self.idx += 1
        plt.clf()

    @classmethod
    def plot_final(cls, args, representative_points):
        x = [p.coords[0] for p in representative_points]
        y = [p.coords[1] for p in representative_points]
        w = [p.weight for p in representative_points]
        plt.xlim(args.center_limit)
        plt.ylim(args.center_limit)
        plt.scatter(x, y, s=w)
        plt.savefig(args.output + "final_representative_points.png")
        plt.clf()

    @classmethod
    def plot_initial_clusters(cls, args, blob_points, cluster):
        x = [p[0] for p in blob_points]
        y = [p[1] for p in blob_points]
        plt.xlim(args.center_limit)
        plt.ylim(args.center_limit)
        plt.scatter(x, y, c=cluster)
        plt.savefig(args.output + "initial_clusters.png")
        plt.clf()

