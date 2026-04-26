#pragma once
#include "tensor.h"
#include "noise_schedule.h"

void q_sample(
    Tensor& xt,
    const Tensor& x0, const Tensor& eps,
    const int* timesteps_device,
    const NoiseSchedule& sched);

void mse_loss_forward(Tensor& loss, const Tensor& pred, const Tensor& target);
void mse_loss_backward(Tensor& dpred, const Tensor& pred, const Tensor& target);

void ddpm_step(
    Tensor& x_prev,
    const Tensor& x_t, const Tensor& x_0_hat, const Tensor& z,
    int t, const NoiseSchedule& sched);

void ddim_step(
    Tensor& x_prev,
    const Tensor& x_t, const Tensor& x_0_hat,
    int t, int t_prev,
    const NoiseSchedule& sched);