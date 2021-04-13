# download best intent model
mkdir -p best_model/intent/init
wget https://www.dropbox.com/s/7id11bznvky7j1w/046-0.2837-0.9273.pt?dl=1 -O best_model/intent/init/046-0.2837-0.9273.pt

# download best slot model
mkdir -p best_model/slot/init_add_labels
wget https://www.dropbox.com/s/nrwb6hdg76d15u7/025-0.0318-0.7820.pt?dl=1 -O best_model/slot/init_add_labels/025-0.0318-0.7870.pt