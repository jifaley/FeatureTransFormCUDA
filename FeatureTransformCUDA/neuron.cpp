#include "neuron.h"

// Define a function to convert a list of neuron nodes to SWC format string
std::string to_swc(const std::vector<NeuronNode>& nodes) {
	// Initialize an empty string to store the result
	std::string swc = "";

	// Loop over the nodes and append each node's information to the string
	for (const NeuronNode& node : nodes) {
		swc += std::to_string(node.id) + " "; // The node id
		swc += std::to_string(node.color) + " "; // The node type (0 for undefined)
		swc += std::to_string(node.x) + " "; // The x coordinate
		swc += std::to_string(node.y) + " "; // The y coordinate
		swc += std::to_string(node.z) + " "; // The z coordinate
		swc += std::to_string(node.r) + " "; // The radius
		swc += std::to_string(node.pid) + "\n"; // The parent node id
	}

	// Return the string
	return swc;
}

std::vector<NeuronNode> readSWCFile(const std::string& filename, float x_mod, float y_mod, float z_mod) {
	std::vector<NeuronNode> nodes;

	std::ifstream infile(filename);
	if (!infile) {
		std::cerr << "无法打开文件: " << filename << std::endl;
		return nodes;
	}

	std::string line;
	while (std::getline(infile, line)) {
		// 忽略注释行  
		if (line[0] == '#') continue;

		std::istringstream iss(line);
		NeuronNode node;
		if (!(iss >> node.id >> node.color >> node.x >> node.y >> node.z >> node.r >> node.pid)) {
			std::cerr << "读取文件时出错: " << filename << std::endl;
			break;
		}

		node.x *= x_mod;
		node.y *= y_mod;
		node.z *= z_mod;

		node.r *= sqrt(x_mod* x_mod + y_mod * y_mod + z_mod * z_mod);

		nodes.push_back(node);
	}

	return nodes;
}