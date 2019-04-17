# directory to data
DATA_DIR=./data
# directory to cache files
TMP_DIR=./tmp
# oxford5k, oxford105k, paris6k, paris106k
DATASET=oxford5k
# resnet or siamac
FEATURE_TYPE=resnet

.PHONY: rank
rank:
	python rank.py \
		--cache_dir $(TMP_DIR)/$(DATASET)_$(FEATURE_TYPE) \
		--query_path $(DATA_DIR)/query/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gallery_path $(DATA_DIR)/gallery/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gnd_path $(DATA_DIR)/gnd_$(DATASET).pkl \
		--dataset_name $(DATASET) \
		--truncation_size 1000


.PHONY: mat2npy
mat2npy:
	python mat2npy.py \
		--dataset_name $(DATASET) \
		--feature_type $(FEATURE_TYPE) \
		--mat_dir $(DATA_DIR)


.PHONY: download
download:
	wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/oxford5k/gnd_oxford5k.pkl -O $(DATA_DIR)/gnd_oxford5k.pkl
	wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/paris6k/gnd_paris6k.pkl -O $(DATA_DIR)/gnd_paris6k.pkl
	ln -s $(DATA_DIR)/gnd_oxford5k.pkl $(DATA_DIR)/gnd_oxford105k.pkl
	ln -s $(DATA_DIR)/gnd_paris6k.pkl $(DATA_DIR)/gnd_paris106k.pkl
	for dataset in oxford5k oxford105k paris6k paris106k; do \
		for feature in siamac resnet; do \
			wget ftp://ftp.irisa.fr/local/texmex/corpus/diffusion/data/$$dataset\_$$feature.mat -O $(DATA_DIR)/$$dataset\_$$feature.mat; \
		done; \
	done
