#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;

REGISTER_OP("QueryAndInterpolation")
	.Input("xyz1: float32")//(b,n,3)
	.Input("xyz2: float32")//(b,npoint,3)
	.Input("lh: float32")
	.Output("weight_space:float32")//(b,npoint,27,3)
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
		c->WithRank(c->input(1), 3, &dims2);
		int num_of_space = 27;
		::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({ c->Dim(dims2, 0), c->Dim(dims2, 1), num_of_space, c->Dim(dims2, 2) });
		c->set_output(0, output);
		return Status::OK();
	});
REGISTER_OP("QueryAndInterpolationGrad")
	.Input("xyz1: float32")//(b,n,3)
	.Input("xyz2: float32")//(b,npoint,3)
	.Input("lh: float32")
	.Input("grad_out:float32")//(b.npoint,3)
	.Output("grad_points1:float32")
	.Output("grad_points2:float32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		c->set_output(1, c->input(1));
		return Status::OK();
	});

void queryAndInterpolationLauncher(int b, int n, int m, int c, const float *xyz1, const float *xyz2, const float *lh, float *weight_space);
class QueryAndInterpolationGpuOp : public OpKernel {
public:
	explicit QueryAndInterpolationGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		
		const Tensor& xyz1_tensor = context->input(0);
		OP_REQUIRES(context, xyz1_tensor.dims() == 3 && xyz1_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
		int b = xyz1_tensor.shape().dim_size(0);
		int n = xyz1_tensor.shape().dim_size(1);
		int c = xyz1_tensor.shape().dim_size(2);

		const Tensor& xyz2_tensor = context->input(1);
		OP_REQUIRES(context, xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
		int m = xyz2_tensor.shape().dim_size(1);
		
		const Tensor& lh_tensor = context->input(2);

		int num_of_space = 27;
		Tensor *weight_space_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,num_of_space,c}, &weight_space_tensor));

		auto xyz1_flat = xyz1_tensor.flat<float>();
		const float *xyz1 = &(xyz1_flat(0));
		auto xyz2_flat = xyz2_tensor.flat<float>();
		const float *xyz2 = &(xyz2_flat(0));
		auto lh_flat = lh_tensor.flat<float>();
		const float *lh = &(lh_flat(0));
		auto weight_space_flat = weight_space_tensor->flat<float>();
		float *weight_space = &(weight_space_flat(0));
		queryAndInterpolationLauncher(b, n, m, c, xyz1, xyz2, lh, weight_space);
	}
};
REGISTER_KERNEL_BUILDER(Name("QueryAndInterpolation").Device(DEVICE_GPU), QueryAndInterpolationGpuOp);

void queryAndInterpolationGradLauncher(int b, int n, int m, int c, const float *xyz1, const float *xyz2, const float *lh,const float *grad_out, float *grad_points1, float *grad_points2);
class QueryAndInterpolationGradGpuOp : public OpKernel {
public:
	explicit QueryAndInterpolationGradGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		const Tensor& xyz1_tensor = context->input(0);
		OP_REQUIRES(context, xyz1_tensor.dims() == 3 && xyz1_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
		int b = xyz1_tensor.shape().dim_size(0);
		int n = xyz1_tensor.shape().dim_size(1);
		int c = xyz1_tensor.shape().dim_size(2);

		const Tensor& xyz2_tensor = context->input(1);
		OP_REQUIRES(context, xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
		int m = xyz2_tensor.shape().dim_size(1);

		const Tensor& lh_tensor = context->input(2);

		const Tensor& grad_out_tensor = context->input(3);

		Tensor *grad_points1_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{ b,n,c }, &grad_points1_tensor));

		Tensor *grad_points2_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{ b,m,c }, &grad_points2_tensor));

		auto xyz1_flat = xyz1_tensor.flat<float>();
		const float *xyz1 = &(xyz1_flat(0));
		auto xyz2_flat = xyz2_tensor.flat<float>();
		const float *xyz2 = &(xyz2_flat(0));
		auto lh_flat = lh_tensor.flat<float>();
		const float *lh = &(lh_flat(0));
		auto grad_out_flat = grad_out_tensor.flat<float>();
		const float *grad_out = &(grad_out_flat(0));
		auto grad_points1_flat = grad_points1_tensor->flat<float>();
		float *grad_points1 = &(grad_points1_flat(0));
		auto grad_points2_flat = grad_points2_tensor->flat<float>();
		float *grad_points2 = &(grad_points2_flat(0));
		queryAndInterpolationGradLauncher(b, n, m, c, xyz1, xyz2, lh, grad_out, grad_points1, grad_points2);
	}
};
REGISTER_KERNEL_BUILDER(Name("QueryAndInterpolationGrad").Device(DEVICE_GPU), QueryAndInterpolationGradGpuOp);
