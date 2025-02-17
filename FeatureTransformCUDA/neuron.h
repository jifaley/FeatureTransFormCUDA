#pragma once
#include <string>
#include <vector>
#include <fstream>  
#include <iostream>  
#include <sstream> 


// Define a structure to represent a neuron node
struct NeuronNode {
	int id; // The node id
	int color;
	double x; // The x coordinate
	double y; // The y coordinate
	double z; // The z coordinate
	double r; // The radius
	int pid; // The parent node id

	NeuronNode() 
	{
		this->id = 0;
		this->color = 0;
		this->x = -1;
		this->y = -1;
		this->z = -1;
		this->r = 0;
		this->pid = -1;
	};
	// A constructor to initialize the node with given values
	NeuronNode(int id, int color,  double x, double y, double z, double r, int pid) {
		this->id = id;
		this->color = color;
		this->x = x;
		this->y = y;
		this->z = z;
		this->r = r;
		this->pid = pid;
	}
};

std::string to_swc(const std::vector<NeuronNode>& nodes);


std::vector<NeuronNode> readSWCFile(const std::string& filename, float x_mod = 1, float y_mod = 1, float z_mod = 1);