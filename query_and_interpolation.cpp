// input: L(1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: weight_space(b,m,27,3)
void query_and_interpolation_cpu(int b, int n, int m, int c, float L, const float *xyz1, const float *xyz2, float *weight_space) {
	float alpha = 3 * sqrtf(3) / 2;
	float dist[27] = 0;
	for (int i = 0; i < b; ++i) {
		for (int j = 0; j < m; ++j) {
			float dist_sum[27] = 0;
			int pts = 0;
			for (int k = 0; k < n; ++k) {
				float x2 = xyz2[j * 3 + 0];
				float y2 = xyz2[j * 3 + 1];
				float z2 = xyz2[j * 3 + 2];
				float x1 = xyz1[k * 3 + 0];
				float y1 = xyz1[k * 3 + 1];
				float z1 = xyz1[k * 3 + 2];
				float d = max(sqrtf((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1) + (z2 - z1)*(z2 - z1)), 1e-20f);
				if (d < alpha*L) {
					pts += 1;
					int cnt = 0;
					for (int o = -1; o < 2; ++o) {
						for (int p = -1; p < 2; ++p) {
							for (int q = -1; q < 2; ++q) {
								x2_ = x2 + o * L;
								y2_ = y2 + p * L;
								z2_ = z2 + q * L;
								dist[cnt] = sqrtf((x2_ - x1)*(x2_ - x1) + (y2_ - y1)*(y2_ - y1) + (z2_ - z1)*(z2_ - z1));
								dist_sum[cnt] += dist[cnt];
								weight_space[j * 27 * 3 + cnt * 3 + 0] += x1 / dist[cnt];
								weight_space[j * 27 * 3 + cnt * 3 + 1] += y1 / dist[cnt];
								weight_space[j * 27 * 3 + cnt * 3 + 2] += z1 / dist[cnt];
								cnt += 1;
							}
						}
					}
				}
			}
			for (int s = 0; s < 27; ++s) {
				weight_space[j * 27 * 3 + s * 3 + 0] *= (dist_sum[s] / pts);
				weight_space[j * 27 * 3 + s * 3 + 1] *= (dist_sum[s] / pts);
				weight_space[j * 27 * 3 + s * 3 + 2] *= (dist_sum[s] / pts);
			}
		}
	}
	xyz1 += n * 3;
	xyz2 += m * 3;
	weight_space += m * 27 * 3;
}

void query_and_interpolation_grad_cpu(int b, int n, int m, int c, float L, const float *xyz1, const float *xyz2,const float *grad_out, float *grad_points) {
	grad_points = grad_out;
}