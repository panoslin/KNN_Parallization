all: serial threaded openmp mpi
serial: serial.cpp
	g++ -std=c++11 -o serial serial.cpp -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
threaded: threaded.cpp
	g++ -pthread -std=c++20 -o threaded threaded.cpp -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
openmp: openmp.cpp
	g++ -fopenmp -std=c++20 -o openmp openmp.cpp -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
mpi: mpi.cpp
	mpicxx -std=c++20 -o mpi mpi.cpp -I/usr/lib/openmpi/include -L/usr/lib/openmpi/lib -lmpi -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
clean:
	rm serial threaded openmp mpi