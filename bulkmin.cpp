#include <stdlib.h>
#include <stdio.h>
#include <arrayfire.h>
#include <iostream>

#include "WolframLibrary.h"

#define N 3

af::array potential(af::array phi, af::array a, af::array phiext, af::array gamma) {
    af::array r1 = -2*a*af::cos(phi) - 2*N*af::cos((phiext-phi)/N);
    af::array r2 = a*af::sin(phi) + af::sin((phi-phiext)/N);
    af::array r3 = r1 + gamma*(r2*r2);

    return r3;
}

af::array minimize_energy(int samplePoints, af::array a, af::array phiext, af::array gamma){

#define NPOINTS 16*N
#define PHINC   af::Pi/8

    // Rough Sweep
    af::array phi     = af::iota(af::dim4(NPOINTS)) * PHINC;

    af::array phi_mod = af::tile(af::moddims(phi, 1, NPOINTS), samplePoints, 1);

    af::array a_mod       = af::tile(a, 1, NPOINTS);
    af::array phiext_mod  = af::tile(phiext, 1, NPOINTS);
    af::array gamma_mod   = af::tile(gamma, 1, NPOINTS);

    af::array q = potential(phi_mod, a_mod, phiext_mod, gamma_mod);

    af::array min_idxs = af::constant(0, af::dim4(samplePoints), s32);
    af::array min_vals = af::constant(0, af::dim4(samplePoints));
    af::min(min_vals, min_idxs, q, 1);

    af::array midpt = phi(min_idxs);

    // Golden Section Search over bracketing interval [a,b]
    
    double gr = 1.61803;

    af::array A = midpt - PHINC;
    af::array B = midpt + PHINC;

    af::array C = B - (B-A)/gr;
    af::array D = A + (B-A)/gr;

    float err;

    do{
        af::array lo = potential(C, a, phiext, gamma);
        af::array hi = potential(D, a, phiext, gamma);

        gfor(af::seq i, samplePoints){
           af::array cond = (lo(i) < hi(i));

           B(i) = (!cond).as(f32)*B(i) + (cond).as(f32)*D(i);
           A(i) = (cond).as(f32)*A(i) + (!cond).as(f32)*C(i);
        }

        C = B - (B-A)/gr;
        D = A + (B-A)/gr;

        float *temp;
        temp = af::min(af::abs(C-D)).host<float>();
        err = temp[0];
        af::freeHost(temp);

    }while(err > 1e-4);

    return (C+D)/2;
}


extern "C" DLLEXPORT int find_minimal_energy_phase(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {

    MTensor a      = MArgument_getMTensor(Args[0]);
    MTensor phiext = MArgument_getMTensor(Args[1]);
    MTensor gamma  = MArgument_getMTensor(Args[2]);

    if(libData->MTensor_getRank(a) != 1) return LIBRARY_RANK_ERROR;
    if(libData->MTensor_getRank(phiext) != 1) return LIBRARY_RANK_ERROR;
    if(libData->MTensor_getRank(gamma) != 1) return LIBRARY_RANK_ERROR;

    mint sample_size = libData->MTensor_getDimensions(a)[0];
    if(sample_size != libData->MTensor_getDimensions(phiext)[0]) return LIBRARY_DIMENSION_ERROR;
    if(sample_size != libData->MTensor_getDimensions(gamma)[0]) return LIBRARY_DIMENSION_ERROR;

    MTensor out;
    mint out_type = MType_Real;
    mint out_rank = 1;
    mint out_dims[1];
    mreal* out_data;
    int err;
    
    out_dims[0] = sample_size;

    af::array aD(sample_size,libData->MTensor_getRealData(a));
    af::array phiextD(sample_size,libData->MTensor_getRealData(phiext));
    af::array gammaD(sample_size,libData->MTensor_getRealData(gamma));

    af::array res = minimize_energy(sample_size, aD, phiextD, gammaD);

    float *temp;
    temp = res.host<float>();

    err = libData->MTensor_new( out_type, out_rank, out_dims, &out );
    if(err){
        af::freeHost(temp);
        return err;
    }

    out_data = libData->MTensor_getRealData(out);

    for(int i=0;i<sample_size;i++) out_data[i] = temp[i];

    af::freeHost(temp);

    MArgument_setMTensor(Res,out);
    return LIBRARY_NO_ERROR;
}
