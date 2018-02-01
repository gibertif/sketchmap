/* Performs out-of-sample sketch-map embedding
   --------------------------------------------------
   Author: Michele Ceriotti, 2011
   Distributed under the GNU General Public License
*/
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "dimreduce.hpp"
#include "../libs/clparser.hpp"
#include "../libs/matrix-io.hpp"

namespace py = pybind11;

using namespace toolbox;

class dimproj{
	private:
		int D, d;
		FMatrix<double> HP,lp;
		bool fweight=false, fdot=false;
		int cgsteps=0;
		double gtemp=0.0,lambda=-1.0;
		double peri=0.0,speri=0.0;
		std::vector<double> pweight;

		double simga_d,sigma_D;
		double a_d,a_D;
		double b_d,b_D;
     	
		NLDRProjection nlproj; NLDROptions opts;
		NLDRMetricPBC nperi; NLDRMetricEuclid neuclid;
		NLDRMetricSphere nsphere; NLDRMetricDot ndot;
	public:
		bool fsimil = false, fprint = false;
		dimproj(int,int,std::vector< std::vector<double> >,std::vector< std::vector<double> >,int,bool);
		~dimproj();
		void set_switch_funcs(std::string ,std::string );
		void set_periodicity(double,double,bool);
		void min_opts(double,double);
		void set_grid(double,int,int);
		void set_switch_funcs();
		void record_opt();
};


void banner()
{
    std::cerr
            << " USAGE: dimproj -D hi-dim -d low-dim -P hd-file -p ld-file [-pi period] [-dot]  \n"
            << "               [-w] [-grid gw,g1,g2 ] [-cgmin st] [-gt temp] [-path lambda]     \n"
            << "           [-fun-hd s,a,b] [-fun-ld s,a,b] [-h] [-print] [-similarity]  < input \n"
            << "                                                                                \n"
            << " computes the projection of the points given in input, given landmark points.   \n"
            << " dimension is set by -D option, and the projection is performed down to the     \n"
            << " dimensionality specified by -d. Optionally, high-dimensional data may be       \n"
            << " assumed to lie in a hypertoroidal space with period -pi. -dot uses a scalar    \n"
            << " product based distance.                                                        \n"
            << " A global minimization is performed on a grid ranging between -gw and +gw in d  \n"
            << " dimensions. First, the stress function is computed on g1 points per dim, then  \n"
            << " an interpolated grid with g2 points is evaluated, and the min selected.        \n"
            << " If -path l is used, path-like averaging is used rather than sketch map.        \n"
            << " If -gt is used, an exponential averaging is used instead of the min.           \n"
            << " If -cgmin is specified, the global minimization is followed by st steps of     \n"
            << " conjugate gradient minimization.                                               \n"
            << " Optionally, a sigmoid function can be applied in the D-dim space (-fun-hd)     \n"
            << " and/or in the d-dim space (-fun-ld). Both parameters must be followed by three \n"
            << " comma separated reals corresponding to sigma, a, b.                            \n"
            << " Data must be provided in the files set by -P and -p and in input in the form:  \n"
            << " X1_1, X1_2, ... X1_D                                                           \n"
            << " X2_1, X2_2, ... X2_D                                                           \n"
            << " If [-w] is present, an extra element per row is expected in the high-dimension \n"
            << " file, which is meant to be the weighting factor for the landmark point.        \n"
            << " -print specifies that the stress functions for each point must be printed out  \n";
}


dimproj::dimproj(int _D,int _d,
		std::vector< std::vector<double> > _listP,
		std::vector< std::vector<double> > _listp,
		int _cgsteps=0,
		bool _fweight=false
		)
{
	// Dimensions
	D    = _D    ;
	d    = _d    ;

	// Store the high dimensional Landmarks
	HP.resize(_listP.size(), D);
	for (unsigned long i=0; i<_listP.size(); ++i) for (unsigned long j=0; j<D; ++j) HP(i,j)=_listP[i][j];

	// Store the low dimensiona Landmarks
	lp.resize(_listp.size(), d);
	for (unsigned long i=0; i<_listp.size(); ++i) for (unsigned long j=0; j<d; ++j) lp(i,j)=_listp[i][j];

	// HP and LP need to match.
	if ((lp.rows())!=HP.rows()) ERROR("HD and LD point list mismatch");

	// storing temperature and conjugate gradient
	opts.cgsteps=_cgsteps;

	// if the flag for the weight is False, all the points are set to 1.0
	fweight=_fweight;
	if (!_fweight) std::fill (pweight.begin(),pweight.end(),1.0);
}

void dimproj::set_periodicity(
		double _peri,
		double _speri,
		bool _fdot=false
		)
{
	// resizing the peridicity and spherical periodicity  
	peri=_peri;
	speri=_speri;
	fdot=_fdot;

	nperi.periods.resize(D); nperi.periods=peri;
	nsphere.periods.resize(D); nsphere.periods=speri;
	
	// periodic options in case we are dealing with a periodic condition
	if (peri==0.0 && speri==0.0) opts.nopts.ometric=&neuclid;
	else if (speri==0) { opts.nopts.ometric=&nperi; }
	else { opts.nopts.ometric=&nsphere; std::cerr<<"Spherical geodesic distances\n"; }
	
	// Check if it is possible to use dot product in the current periodicity
	/*
	if (fdot) 
	{
		std::cerr<<"Using dot product distance!\n";
		if (peri!=0 || speri!=0) ERROR("Cannot use periodic options together with dot product distance.");
		opts.nopts.ometric=&ndot;
	}
	*/
}

void dimproj::min_opts(
		double _gtemp, 
		double _lambda
		)
{
	gtemp=_gtemp;
	lambda=_lambda;
	opts.gtemp=gtemp; 

}

void dimproj::set_grid(double _width, int _gridx, int _gridy)
{
	// Storing the grid, a-la Mik. This sucks, we should implement a set_grid command.
	opts.grid1=int(_gridx); 
	opts.grid2=int(_gridy); 
	opts.gwidth=double(_width); 
}


void dimproj::set_switch_funcs(std::string fdhd,std::string fdld)
{
	std::valarray<double> tfpars;
	std::valarray<double> fhdpars(0.0,3), fldpars(0.0,3), fgrid(0.0,3);

	if (fdld=="identity")
	{ tfpars.resize(0); opts.tfunL.set_mode(NLDRIdentity,tfpars); }
	else
	{
		csv2floats(fdld,tfpars); fldpars=tfpars;
		std::cerr<<"lo-dim pars"<<tfpars<<"\n";
		if (tfpars.size()==2)
		{
			opts.tfunL.set_mode(NLDRGamma,tfpars);
		}
		else if (tfpars.size()==3)
		{
			opts.tfunL.set_mode(NLDRXSigmoid,tfpars);
		}
		else
		{  ERROR("-fun-ld argument must be of the form sigma,a,b or sigma,n");  }
	}
	if (fdhd=="identity")
	{ tfpars.resize(0); opts.tfunH.set_mode(NLDRIdentity,tfpars); }
	else
	{
		csv2floats(fdhd,tfpars);  fhdpars=tfpars;
		std::cerr<<"high-dim pars"<<tfpars<<"\n";
		if (tfpars.size()==2)
		{
			opts.tfunH.set_mode(NLDRGamma,tfpars);
		}
		else if (tfpars.size()==3)
		{
			opts.tfunH.set_mode(NLDRXSigmoid,tfpars);
		}
		else
		{  ERROR("-fun-hd argument must be of the form sigma,a,b or sigma,n");  }
	}
	
}

void dimproj::record_opt()
{
	// set the options for the dim red 
	nlproj.set_options(opts);
}

/*
str::vector< std::vector<double> > dimproj::project(std::vector< std::vector<double> > plist,std::vector< double > pweight)
{
	// set the weight array
	std::valarray<double> nw(n); for (int i=0; i<n; i++)nw[i]=pweight[i];
	// set the points and weights for the dimensionality reduction
	nlproj.set_points(HP,lp,nw);
	// set 
	std::valarray<double> NP(D), PP(D), pp(d), np(d);
	// Set precision for printing
	std::cout.precision(12); std::cout.setf(std::ios::scientific);
	unsigned long ip=0;

	// This is not desired. What we would like is to change the parsing and store the result in 
	// an array containing the points and the errors and the distance, as it is way better.
	// So to do this we just need to loop over the point and push them in an array.
	//
	// For now, we can try to check everything in the set up before trying to project points.
	while (std::cin.good())
	{
		std::cerr<<"Projecting "<<ip++<<"\n";
		for (int i=0; i<D; i++) std::cin>>NP[i];
		
		if (! std::cin.good()) break;
		double mind, perr, w, tw;
		if (fprint) nlproj.interp_out=std::string("interpolant.")+int2str(ip);
		
		if (lambda>0)
		// does path-like interpolation
		{
			pp=0.0; tw=0.0;
			for (unsigned long i=0; i<n; i++)
			{
				w=exp(- opts.nopts.ometric->dist(&NP[0], &HP(i,0), D)/lambda);
				np=lp.row(i); np*=w;
				pp+=np;
				tw+=w;
			}
			pp*=1.0/tw;
			for (int i=0; i<d; i++) std::cout<<pp[i]<<" "; std::cout<<"\n";
		}
		else
		{
			perr=nlproj.project(NP, PP, pp, mind, fsimil);
			for (int i=0; i<d; i++) std::cout<<pp[i]<<" "; std::cout<<perr<<" "<<mind<<"\n";
		}
	}
}
*/

dimproj::~dimproj(){
	nlproj.~NLDRProjection() ; 
	opts.~NLDROptions()    ;
	nperi.~NLDRMetricPBC()   ;  
	neuclid.~NLDRMetricEuclid() ;
	nsphere.~NLDRMetricSphere() ; 
	ndot.~NLDRMetricDot()    ;

}


// exposing functions to python 
void mod_dimproj(py::module &pymod){
	py::class_<dimproj>(pymod,"dimproj",R"sketchmap(
Dimproj
---------------------
)sketchmap")
		.def(py::init<int,int,std::vector< std::vector<double> >,std::vector< std::vector<double> >,int,bool>())
	;
}
