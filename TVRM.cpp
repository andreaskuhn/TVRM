/*
 * TVRM.cpp
 *
 *  Created on: 13.06.2016
 *      Author: andreask
 */

#include <iostream>

#include "ctime"

#include <opencv/cv.h>
#include <opencv/highgui.h>

int width;
int height;
int maxDisp;
cv::Mat img1;
cv::Mat img2;
cv::Mat res;
cv::Mat qual;
int windowsize;
//bool ***costs;
std::vector<std::vector<std::vector<double> > > disps;
std::vector<std::vector<std::vector<double> > > TVs;

void
initialize(){

	width = img1.cols;
	height = img1.rows;
	windowsize = 4;

	disps.resize(width);
//#pragma omp parallel for
	for (int i = 0; i < width; i++) {
		disps[i].resize(height);
		for (int j = 0; j < height; j++){
			disps[i][j].resize(maxDisp);
			for (int k = 0; k < maxDisp; k++){
				disps[i][j][k]=false;
			}
		}
	}

	TVs.resize(width);
//#pragma omp parallel for
	for (int i = 0; i < width; i++) {
		TVs[i].resize(height);
		for (int j = 0; j < height; j++){
			TVs[i][j].resize(maxDisp);
			for (int k = 0; k < maxDisp; k++){
				TVs[i][j][k]=false;
			}
		}
	}

	res = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	qual = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
}

int getIntensCost(int x_l, int y, int x_r){
	return fabs(img1.at<uchar>(y,x_l)-img2.at<uchar>(y,x_r));
}

int getSADCost(int x_l, int y, int x_r){
	int cost=0;
	if(x_l<windowsize || x_r<windowsize || y<windowsize || x_l>img1.cols-windowsize || x_r>img1.cols-windowsize || y>img1.rows-windowsize) return std::numeric_limits<int>::max();
	for(int i=-windowsize; i<=windowsize; i++){
		for(int j=-windowsize; j<=windowsize; j++){
			cost += fabs(img1.at<uchar>(y+j,x_l+i)-img2.at<uchar>(y+j,x_r+i)); //ABS
			//cost += pow(img1.at<uchar>(y+j,x_l+i)-img2.at<uchar>(y+j,x_r+i),2); //SQUARED
		}
	}
	return cost;
}

int
getCensusCost(int x_l, int y, int x_r){
	int cost=0;
	if(x_l<windowsize || x_r<windowsize || y<windowsize || x_l>=img1.cols-windowsize || x_r>=img1.cols-windowsize || y>=img1.rows-windowsize)
		return std::numeric_limits<int>::max();
	for(int i=-windowsize; i<=windowsize; i++){
		for(int j=-windowsize; j<=windowsize; j++){
			if(img1.at<uchar>(y+j,x_l+i)<img1.at<uchar>(y,x_l) && img2.at<uchar>(y+j,x_r+i)>=img2.at<uchar>(y,x_r))
				cost++;
			if(img1.at<uchar>(y+j,x_l+i)>=img1.at<uchar>(y,x_l) && img2.at<uchar>(y+j,x_r+i)<img2.at<uchar>(y,x_r))
				cost++;
		}
	}
	return cost;
}

double
getTVDispD(int i, int j, int disp){

	int dist = 0;
	if(disp<0 || disp>=maxDisp || i<0 || j<0 || i>=width || j>=height)
		return -1.0;

	while(disp-dist>=0 && disp+dist<maxDisp && dist<5){
		if(disps[i][j][disp-dist] != -1.0 || disps[i][j][disp+dist] != -1.0){
			if(disps[i][j][disp-dist] != -1.0){
				//std::cout << "1: " << tmp[i][j][disp-dist] << std::endl;
				return disps[i][j][disp-dist];
			}
			else{
				//std::cout << "2: " << tmp[i][j][disp+dist] << std::endl;
				return disps[i][j][disp+dist];
			}
		}
		dist++;
	}
	return -1.0;
}

double
getTVDispTV(int i, int j, int disp){

	int dist = 0;
	if(disp<0 || disp>=maxDisp || i<0 || j<0 || i>=width || j>=height)
		return 100;

	while(disp-dist>=0 && disp+dist<maxDisp && dist<5){
		if(TVs[i][j][disp-dist] != -1.0 || TVs[i][j][disp+dist] != -1.0){
			if(TVs[i][j][disp-dist] != -1.0){
				//std::cout << "1: " << tmp[i][j][disp-dist] << std::endl;
				return TVs[i][j][disp-dist];
			}
			else{
				//std::cout << "2: " << tmp[i][j][disp+dist] << std::endl;
				return TVs[i][j][disp+dist];
			}
		}
		dist++;
	}
	return 100;
}

void
costCALC(){

#pragma omp parallel for

	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){

			//std::cout << disps[x][y].size() << " " << maxDisp << std::endl;

			disps[x][y][0]=-1.0;
			disps[x][y][maxDisp-1]=-1.0;

			int nCosts[3];
			nCosts[0] = getSADCost(x, y, x);
			nCosts[1] = getSADCost(x, y, x-1);
			for(int k=2; k<maxDisp; k++){

				nCosts[2] = getSADCost(x, y, x-k);

				if(nCosts[1]<nCosts[0] && nCosts[1]<nCosts[2]){
					double disp = (((double)k)*1.0/(double)nCosts[2] + (double)(k-1)*1.0/(double)nCosts[1] + (double)(k-2)*1.0/(double)nCosts[0])
							/ (1.0/(double)nCosts[0] + 1.0/(double)nCosts[1] + 1.0/(double)nCosts[2]);
					disps[x][y][k-1]=disp;

					if(disp<0)
						std::cout << disps[x][y][k-1] << std::endl;

				}
				else
					disps[x][y][k-1]=-1.0;

				nCosts[0] = nCosts[1];
				nCosts[1] = nCosts[2];
			}
		}
	}
}

void
getTV2(){

#pragma omp parallel for

	for(int d=0; d<maxDisp; d++){
		TVs[0][0][d]=-1.0;
		TVs[0][height-1][d]=-1.0;
		TVs[width-1][0][d]=-1.0;
		TVs[width-1][height-1][d]=-1.0;
	}

	for(int y=1; y<height-1; y++){
		for(int x=1; x<width-1; x++){
			for(int d=0; d<maxDisp; d++){
				if(disps[x][y][d]!=-1.0){
					double tv1 = getTVDispD(x , y+1, d) - getTVDispD(x , y-1, d);
					double tv2 = getTVDispD(x+1 , y, d) - getTVDispD(x-1 , y, d);
					if(tv1!=-1 && tv2!=-1){
						TVs[x][y][d]=sqrt(tv1*tv1 + tv2*tv2);
						//TVs[x][y][d]=tv1;
						//if(tv2<tv1) TVs[x][y][d]=tv2;
					}
					else
						TVs[x][y][d]=-1;

					//std::cout << TVs[x][y][d] << " " << tv1 << " " << tv2 << std::endl;
				}
				else
					TVs[x][y][d]=-1.0;
			}
		}
	}
}

void
costAGG(){

#pragma omp parallel for

	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			int max = 0;
			double minCost = 10000;
			int bestDisp = 0;

			for(int d=0; d<maxDisp; d++){

				if(TVs[x][y][d]==-1.0)
					continue;

				//double disp= costs[x][y][d];

				double totalV = 0;
				double border = 1.0;

				int maxNeighb = 50;
				int k = 0;
				int endCount=0;

				//std::cout << std::endl << std::endl;
				while(totalV<border && k<maxNeighb){
					//std::cout << std::endl;
					//std::cout << "-" << std::endl;
					int l=x-k;
					int m=y-k;
					std::vector<double> totalV_(k*8);
					if(k==0) totalV_.resize(1);
					int count =0;
					for(; m<=y+k; m++){
						//std::cout << fabs(getTVDisp(l,m,d)-disp) << " " <<  disp << std::endl;
						totalV_[count++]=(fabs(getTVDispTV(l,m,d)));
						//if(((totalV+totalV_)>border)  && (endCount==0))
						//	endCount=count;
					}
					l=x-k+1;
					m=y+k;
					for(; l<x+k; l++){
						totalV_[count++]=(fabs(getTVDispTV(l,m,d)));
						//if(((totalV+totalV_)>border)  && (endCount==0))
						//	endCount=count;
					}
					l=x-k+1;
					m=y-k;
					for(; l<=x+k; l++){
						totalV_[count++]=(fabs(getTVDispTV(l,m,d)));
						//if(((totalV+totalV_)>border)  && (endCount==0))
						//	endCount=count;
					}
					l=x+k;
					m=y-k+1;
					for(; m<=y+k; m++){
						totalV_[count++]=(fabs(getTVDispTV(l,m,d)));
						//if(((totalV+totalV_)>border) && (endCount==0))
						//	endCount=count;
					}

					std::sort(totalV_.begin(), totalV_.end());
					totalV = totalV_[totalV_.size()/2];

					//totalV += totalV_ / (double)count;

					for(int ii=0; ii<totalV_.size(); ii++){
						//std::cout << totalV_[ii] << " ";
					}
					//std::cout << std::endl;
					//std::cout << "t: " << totalV << " " << k << " "  << count  << std::endl;
					k++;
				}

				//std::cout << k << std::endl;

				if (k>max){
					bestDisp = d;
					max = k;
					minCost = getCensusCost(x, y, x-d);
				}
				else if(k==max){
					if(getCensusCost(x, y, x-d)<minCost){
						bestDisp = d;
						max = k;
						minCost = getCensusCost(x, y, x-d);
					}
				}
			}
			res.at<uchar>(y,x) = (unsigned char)bestDisp;
			//std::cout << max << std::endl;
			qual.at<uchar>(y,x) = (unsigned char)max*10;
		}
	}
}

void
medianFilter(){

#pragma omp parallel for

	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			int n = (20 - qual.at<uchar>(y,x) / 10) / 2;
			if(n==0)
				n=1;
			if(n>10)
				n=10;
			std::vector<int> dd((2*n+1)*(2*n+1));
			int counter=0;
			for(int i=-n; i<=n; i++){
				for(int j=-n; j<=n; j++){
					if(x+i >=0 && y+j>=0 && x+i < width && y+j < height){
						dd[counter++] = res.at<uchar>(y+j,x+i);
					}
				}
			}
			std::sort(dd.begin(), dd.end());
			res.at<uchar>(y,x) = (unsigned char)dd[dd.size()/2];
		}
	}
}

int main(){

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Adirondack/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Adirondack/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Adirondack/im0D.png";
	std::string qualN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Adirondack/im0Q.png"; */

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/ArtL/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/ArtL/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/ArtL/im0D.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Jadeplant/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Jadeplant/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Jadeplant/im0D.png";
	std::string qualN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Jadeplant/im0Q.png"; */

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Motorcycle/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Motorcycle/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Motorcycle/im0D.png";
	std::string qualN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Motorcycle/im0Q.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/MotorcycleE/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/MotorcycleE/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/MotorcycleE/im0D.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Piano/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Piano/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Piano/im0D.png";
	std::string qualN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Piano/im0Q.png"; */

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Pipes/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Pipes/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Pipes/im0D.png";
	std::string qualN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Pipes/im0Q.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Playroom/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Playroom/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Playroom/im0D.png"; */

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Playtable/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Playtable/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Playtable/im0D.png";*/

	std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/PlaytableP/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/PlaytableP/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/PlaytableP/im0D.png";
	std::string qualN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/PlaytableP/im0Q.png";

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Shelves/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Shelves/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Shelves/im0D.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Teddy/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Teddy/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Teddy/im0D.png";
	std::string qualN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Teddy/im0Q.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Vintage/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Vintage/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/trainingH/Vintage/im0D.png";*/


	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Australia/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Australia/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Australia/im0D.png";  */

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/AustraliaP/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/AustraliaP/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/AustraliaP/im0D.png"; */

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Bicycle2/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Bicycle2/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Bicycle2/im0D.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Classroom2E/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Classroom2E/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Classroom2E/im0D.png"; */

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Computer/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Computer/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Computer/im0D.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Djembe/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Djembe/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Djembe/im0D.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Hoops/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Hoops/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Hoops/im0D.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Newkuba/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Newkuba/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Newkuba/im0D.png";*/

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Plants/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Plants/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Plants/im0D.png"; */

	/*std::string img1N = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Staircase/im0.png";
	std::string img2N =  "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Staircase/im1.png";
	std::string resN = "/net/NAS_S/Bildsequenzen/Middlebury_Stereo/2014_half/MiddEval3/testH/Staircase/im0D.png";*/





	maxDisp = 170;

/*	std::string img1N = "/home/andreask/Bilder/MB_Test/im0_.png";
	std::string img2N =  "/home/andreask/Bilder/MB_Test/im1_.png";
	maxDisp = 70; */

	std::time_t start, end;
	long delta = 0;
	start = std::time(NULL);

	// do your code here

	end = std::time(NULL);

	img1 = cvLoadImage(&img1N[0], CV_LOAD_IMAGE_GRAYSCALE);
	img2 = cvLoadImage(&img2N[0], CV_LOAD_IMAGE_GRAYSCALE);

	std::cout << std::endl << "init" << std::endl;
	clock_t start2 = std::time(NULL);
	initialize();
	clock_t end2 = std::time(NULL);
	std::cout << end2 - start2 << std::endl;

	std::cout << std::endl << "CALC" << std::endl;
	start2 = std::time(NULL);
	costCALC();
	end2 = std::time(NULL);
	std::cout << end2 - start2 << std::endl;

	std::cout << std::endl << "TV2" << std::endl;
	start2 = std::time(NULL);
	getTV2();
	end2 = std::time(NULL);
	std::cout << end2 - start2 << std::endl;

	std::cout << std::endl << "AGGR" << std::endl;
	start2 = std::time(NULL);
	costAGG();
	end2 = std::time(NULL);
	std::cout << end2 - start2 << std::endl;

	std::cout << std::endl << "Median" << std::endl;
	start2 = std::time(NULL);
	//cv::medianBlur(res, res, 3);
	//medianFilter();
	end2 = std::time(NULL);
	std::cout << end2 - start2 << std::endl;

	cv::imwrite(&resN[0], res);
	cv::imwrite(&qualN[0], qual);

	end = std::time(NULL);
	std::cout << std::endl << "overall: " << end - start << std::endl;

}
