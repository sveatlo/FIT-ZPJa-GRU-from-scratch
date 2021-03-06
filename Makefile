.PHONY=run all build pack docs banner

# build variables
CFLAGS=-std=c++11 -lstdc++ -lm -Wall -Wextra -O3
CC=gcc
## which modules should be build
MODULES=log matrix gru
OBJECT_FILE_PATTERN=$(DIST_DIR)%.o
SRC_DIR=src/
DIST_DIR=dist/
DOCS_DIR=docs/
BINARY_NAME=gru-nn
ARCHIVEFILENAME=xhanze10.tar.gz

# documentation variables
DOCS_SOURCES=$(DOCS_DIR)manual/documentation.tex $(DOCS_DIR)manual/czechiso.bst \
	$(DOCS_DIR)manual/references.bib $(DOCS_DIR)manual/Makefile $(DOCS_DIR)manual/images
PDF_FILENAME=report.pdf

all: build $(DIST_DIR)$(BINARY_NAME) banner

banner:
	@echo " ██████╗ ██████╗ ██╗   ██╗    ███╗   ██╗███╗   ██╗"
	@echo "██╔════╝ ██╔══██╗██║   ██║    ████╗  ██║████╗  ██║"
	@echo "██║  ███╗██████╔╝██║   ██║    ██╔██╗ ██║██╔██╗ ██║"
	@echo "██║   ██║██╔══██╗██║   ██║    ██║╚██╗██║██║╚██╗██║"
	@echo "╚██████╔╝██║  ██║╚██████╔╝    ██║ ╚████║██║ ╚████║"
	@echo " ╚═════╝ ╚═╝  ╚═╝ ╚═════╝     ╚═╝  ╚═══╝╚═╝  ╚═══╝"
	@echo " _                                 _   _       "
	@echo "| |__  _   _   _____   _____  __ _| |_| | ___  "
	@echo "| '_ \\| | | | / __\\ \\ / / _ \\/ _\` | __| |/ _ \\ "
	@echo "| |_) | |_| | \\__ \\\\ V /  __/ (_| | |_| | (_) |"
	@echo "|_.__/ \\__, | |___/ \\_/ \\___|\\__,_|\\__|_|\\___/ "
	@echo "       |___/                                   "


dev: build

documentation: $(wildcard $(SRC_DIR)*)
	doxygen
	OUTPUT_PDF=$(PDF_FILENAME) make -C $(DOCS_DIR)/report

stats:
	@echo -n "Lines of code: " && wc -l $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h) | tail -n 1 | sed -r "s/[ ]*([0-9]+).*/\1/g"
	@echo -n "Size of code: " && du -hsc $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h) | tail -n 1 | cut -f 1

$(DIST_DIR):
	mkdir -p $(DIST_DIR)

# Link all the modules together
build: $(DIST_DIR) $(DIST_DIR)$(BINARY_NAME)

build-prod: build
	mv $(DIST_DIR)$(BINARY_NAME) ./$(BINARY_NAME)

# Build binary
$(DIST_DIR)$(BINARY_NAME): $(SRC_DIR)main.cpp $(patsubst %,$(OBJECT_FILE_PATTERN), $(MODULES))
	$(CC) $(CFLAGS) \
		$(SRC_DIR)main.cpp $(patsubst %,$(OBJECT_FILE_PATTERN), $(MODULES)) \
	-o $(DIST_DIR)$(BINARY_NAME)

# Make modules independently
$(OBJECT_FILE_PATTERN): $(SRC_DIR)%.cpp $(SRC_DIR)%.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)$*.cpp -o $(DIST_DIR)$*.o

run: build
	exec $(DIST_DIR)$(BINARY_NAME)
pack: $(SRC_DIR)*.cpp $(SRC_DIR)*.h Makefile Doxyfile documentation
	mv docs/report/$(PDF_FILENAME) $(PDF_FILENAME)
	make clean
	tar zcf $(ARCHIVEFILENAME) $(SRC_DIR) $(DIST_DIR) $(DOCS_DIR) data/ $(PDF_FILENAME) Makefile Doxyfile README.md
clean:
	make -C $(DOCS_DIR)/report clean
	rm -rf ./*.o $(DIST_DIR)$(BINARY_NAME) $(DIST_DIR)*.o $(DIST_DIR)*.out $(DIST_DIR)*.a $(DIST_DIR)*.so $(SRC_DIR)*.gch \
			$(ARCHIVEFILENAME) $(DOCS_DIR)doxygen #$(PDF_FILENAME)
