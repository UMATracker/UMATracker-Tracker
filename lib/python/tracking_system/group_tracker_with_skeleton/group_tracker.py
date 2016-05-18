# Tsukasa Fukunaga*, Shoko Kubota, Shoji Oda, and Wataru Iwasaki*. GroupTracker: Video Tracking System for Multiple Animals under Severe Occlusion. Computational Biology and Chemistry 57, 39-45. (2015)

from sklearn import mixture, cluster
import numpy as np
from numpy import linalg as LA

try:
    from .hungarian import Hungarian
except SystemError:
    from hungarian import Hungarian

EPS = np.finfo(float).eps

class GroupTrackerGMM(mixture.GMM):
    # TODO:要調整（というより，UIから調整可能にすること）
    alpha = 0.7
    max_dist = 5
    hungarian = Hungarian()

    def set_likelihood_diff_threshold(self, th):
        self.alpha = th

    def _fit(self, X, y=None, do_prediction=False, n_k_means=None, lost_mode=False):
        if n_k_means is None:
            n_k_means = self.n_components

        if not lost_mode:
            if hasattr(self, 'weights_'):
                self.prev_weights_ = self.weights_.copy()
            if hasattr(self, 'means_'):
                self.prev_means_ = self.means_.copy()
            if hasattr(self, 'covars_'):
                self.prev_covars_ = self.covars_.copy()

        print(self.init_params)
        self.verbose = 2

        if self.init_params != '':
            self.means_ = cluster.KMeans(
                    n_clusters=self.n_components,
                    random_state=self.random_state,
                    max_iter=10000).fit(X).cluster_centers_

        resp = super(GroupTrackerGMM, self)._fit(X, y, do_prediction)

        if self.params == 'wc':
            self.params = 'wmc'

        if self.init_params != '':
            self.lambdas = np.sort(LA.eigvals(self.covars_))
            log_likelihoods, responsibilities = self.score_samples(X)
            self.log_likelihood = log_likelihoods.mean()
            print(self.log_likelihood)
            self.init_params = ''
        else:
            log_likelihoods, responsibilities = self.score_samples(X)
            if not lost_mode and log_likelihoods.mean() - self.log_likelihood < -self.alpha:
            # if not lost_mode and (np.fabs(log_likelihoods.mean() - self.log_likelihood) > self.alpha or np.any(dists>self.max_dist)):
                print('Lost likeli: {0}'.format(log_likelihoods.mean()))
                means = cluster.KMeans(
                        n_clusters=n_k_means,
                        max_iter=10000,
                        random_state=self.random_state).fit(X).cluster_centers_

                prev_means_shape = self.prev_means_.shape
                cost_mtx_shape = (n_k_means, prev_means_shape[0], 2)

                new_means_mtx = np.tile(
                        means,
                        prev_means_shape[0]
                        ).reshape(cost_mtx_shape)

                n_elems_old_means_mtx = prev_means_shape[0] * prev_means_shape[1]
                prev_means_mtx = np.repeat(
                        self.prev_means_.reshape(1, 1, n_elems_old_means_mtx),
                        n_k_means,
                        axis=0
                        ).reshape(cost_mtx_shape)

                cost_mtx = LA.norm(new_means_mtx - prev_means_mtx, axis=2)
                self.hungarian.calculate(cost_mtx)

                for pos in self.hungarian.get_results():
                    self.means_[pos[1], :] = means[pos[0], :]

                self.weights_ = self.prev_weights_
                self.covars_ = self.prev_covars_
                resp = self._fit(X, y, do_prediction, n_k_means, True)

                dists = LA.norm(self.means_, axis=1)
                for i, dist in enumerate(dists):
                    if dist < EPS:
                        self.means_[i, :] = self.prev_means_[i, :]
                        self.covars_[i, :] = self.prev_covars_[i, :]

            dists = LA.norm(self.means_, axis=1)
            for i, dist in enumerate(dists):
                if dist < EPS:
                    self.means_[i, :] = self.prev_means_[i, :]
                    self.covars_[i, :] = self.prev_covars_[i, :]

        log_likelihoods, responsibilities = self.score_samples(X)
        print('End likeli: {0}'.format(log_likelihoods.mean()))
        return resp

    def _do_mstep(self, X, responsibilities, params, min_covar=0):
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        if 'w' in params:
            self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
        if 'm' in params:
            self.means_ = weighted_X_sum * inverse_weights
        if 'c' in params:
            if self.init_params != '':
                covar_mstep_func = mixture.gmm._covar_mstep_funcs[self.covariance_type]
                self.covars_ = covar_mstep_func(
                    self, X, responsibilities, weighted_X_sum, inverse_weights,
                    min_covar)
            else:
                self.covars_ = covar_mstep_custom(
                    self, X, responsibilities, weighted_X_sum, inverse_weights,
                    min_covar)
        return weights


def covar_mstep_custom(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    n_features = X.shape[1]
    cv = np.empty((gmm.n_components, n_features, n_features))
    for c in range(gmm.n_components):
        post = responsibilities[:, c]
        mu = gmm.means_[c]
        diff = X - mu
        diff_mul_xy = np.prod(diff, axis=1)
        diff_sub_xy = -np.diff(np.square(diff)).flatten()
        with np.errstate(under='ignore'):
            # Underflow Errors in doing post * X.T are  not important
            numerator = np.dot(post, diff_mul_xy)
            denominator = np.dot(post, diff_sub_xy)
            if np.fabs(denominator) < EPS:
                theta = np.pi/4.0
            else:
                theta = np.arctan2(2.0 * numerator, denominator)/2.0
                if theta < 0.0:
                    theta += np.pi/2.0;
            likelihood_2nd_diff = -(denominator*np.cos(2.0*theta) + 2.0*numerator*np.sin(2.0*theta))
            if likelihood_2nd_diff >= 0.0:
                theta += np.pi/2.0

            u0 = np.array([-np.sin(theta), np.cos(theta)])
            u1 = np.array([np.cos(theta),  np.sin(theta)])
            avg_cv = gmm.lambdas[c, 0] * np.outer(u0, u0) + gmm.lambdas[c, 1] * np.outer(u1, u1)
        cv[c] = avg_cv + min_covar * np.eye(n_features)
    return cv

if __name__ == '__main__':
    import itertools
    from scipy import linalg
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    # Fit a mixture of Gaussians with EM using five components
    gmm = GroupTrackerGMM(n_components=2, covariance_type='full', n_iter=1000)
    gmm.fit(X)
    X += np.array([0.,-10.])
    gmm.fit(X)

    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

    (clf, title) = (gmm, 'GMM')
    splot = plt.subplot(2, 1, 1)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-10, 10)
    plt.ylim(-10, 6)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

    plt.show()
