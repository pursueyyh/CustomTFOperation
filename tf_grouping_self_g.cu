// input: L (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: weight_space (b,m,27,3)
__global__ void query_and_interpolation_gpu(int b, int n, int m, int c, const float *xyz1, const float *xyz2, const float *lh, float *weight_space) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    weight_space += m*27*3*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    float L=lh[0];
    int pts;
    int cnt;
    float alpha=3*sqrtf(3)/2;
    float dist[27];
    float dist_sum[27];
    float x2_,y2_,z2_;

    for(int j=index;j<m;j+=stride){
	for(int f=0;f<27;++f) dist_sum[f]=0;
	pts=0;
	for(int k=0;k<n;++k){
	    float x2=xyz2[j*3+0];
	    float y2=xyz2[j*3+1];
	    float z2=xyz2[j*3+2];
	    float x1=xyz1[k*3+0];
	    float y1=xyz1[k*3+1];
	    float z1=xyz1[k*3+2];
	    float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
	    if (d<alpha*L) {
                pts += 1;
		cnt = 0;
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

__global__ void query_and_interpolation_grad_gpu(int b, int n, int m, int c, const float *xyz1, const float *xyz2, const float *lh, const float *grad_out, float *grad_points1, float *grad_points2) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    grad_out += m*27*3*batch_index;
    grad_points1 += n*3*batch_index;
    grad_points2 += m*3*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    float L=lh[0];
    int pts;
    int cnt;
    float alpha=3*sqrtf(3)/2;
    float dist[27];
    float dist_sum[27];
    float x2_,y2_,z2_;

}

void queryAndInterpolationLauncher(int b, int n, int m, int c, const float *xyz1, const float *xyz2,const float *lh, float *weight_space) {
    query_and_interpolation_gpu<<<b,256>>>(b,n,m,c,xyz1,xyz2,lh,weight_space);
    //cudaDeviceSynchronize();
}
void queryAndInterpolationGradLauncher(int b, int n, int m, int c, const float *xyz1, const float *xyz2, const float *lh, const float *grad_out, float *grad_points1, float *grad_points2) {
    query_and_interpolation_grad_gpu<<<b,256>>>(b,n,m,c,xyz1,xyz2,lh,grad_out,grad_points1,grad_points2);
    //cudaDeviceSynchronize();
}
