# PDF Demo 

## Tools used

- tensorflow_gpu==2.2.0
- numpy==1.18.5
- requests==2.22.0
- matplotlib==3.3.0
- dataclasses==0.8
- Pillow==8.1.2
- PyPDF2==1.26.0
- PyYAML==5.4.1
- scikit_learn==0.24.1
- tensorflow==2.4.1
- streamlit==0.78.0
- Blender (Not python, used to make the video)
## How to run the streamlit app 

All you need to do is a pip install and then point streamlit at the 
python app file.

```bash
# bash/zsh
pip install streamlit

cd /<where you downloaded it>pdemo/scripts/streamlit

streamlit run ch_streamlit_app.py
```


## Usage - Python

```python
import pdemo as D

### Download data into the .cache in the datasets folder
D.datasets.pdfs.download()

### Extract images from the pdfs in the cache
df = D.datasets.preprocessing_toolbox.extract_images_from_pdfs() 

### Scrub out any that may have had an issue loading
df = df.loc[~df.errors]

### Setup a model build
E = D.neural.model.SetupTrainingParticulars(df = df) 

E.kickoff_training_run()

### Wait for neural net to train 
E.save_model()


##
```

## Pdf Download Config File

```
./pdemo/datasets/_pdfs_metadata.yaml
```

## Directory Topology

```bash
tree
.
├── README.md
├── __init__.py
├── context
│   ├── __init__.py
│   └── _context.py
├── datasets
│   ├── __init__.py
│   ├── _pdf_images.py
│   ├── _pdfs.py
│   ├── _pdfs_metadata.yaml
│   ├── _test_images.py
│   ├── pdf_imgs
│   │   ├── ambulatory_laboratory_products-e888cee8bd3b4ec91815ed168a520934.pdf
│   │   ├── ambulatory_laboratory_products.png
│   │   ├── anatomic_pathology-30509824e3020cde2e92d433866ddf83.pdf
│   │   ├── anatomic_pathology.png
│   │   ├── clinical_chemistry-bd410acd4af7055e6a91435bcf944b4f.pdf
│   │   ├── clinical_chemistry.png
│   │   ├── equipment-176a33f8e631e8f26a1b1a9c63cd3e9b.pdf
│   │   ├── equipment.png
│   │   ├── general_lab-c31479c6495cc0b35ab30ee69e2681ee.pdf
│   │   ├── general_lab.png
│   │   ├── hematology-de2243fea907d16cfd0f5fc3c740c8f7.pdf
│   │   ├── hematology.png
│   │   ├── lab_equipment-62542382d78c6ad64a2e9a4a03cedb64.pdf
│   │   ├── lab_equipment.png
│   │   ├── laboratory_products-b349b3ad57f8ed01f587750d8423df2a.pdf
│   │   ├── laboratory_products.png
│   │   ├── microbiology-b4c8f110ec664ff934827ad01074f6fc.pdf
│   │   ├── microbiology.png
│   │   ├── rapid_diagnostics-051fae7b62bcf3d3f4822e005dc136ed.pdf
│   │   ├── rapid_diagnostics.png
│   │   ├── sharps_safety-cd021e98b0277931fc3973e4ca024c08.pdf
│   │   ├── sharps_safety.png
│   │   ├── specimen_collection-e426036b4cde65546b95d0ceeec6125a.pdf
│   │   └── specimen_collection.png
│   └── test_images_cache
│       ├── box_gloves.jpeg
│       ├── latex_gloves.jpeg
│       └── thumbs_up_glove.jpeg
├── logs
├── neural
│   ├── __init__.py
│   ├── builder
│   │   ├── __init__.py
│   │   └── _builder.py
│   ├── layers
│   │   ├── __init__.py
│   │   └── _data_augmentation_layers.py
│   └── model
│       ├── __init__.py
│       ├── _ops.py
│       ├── label_positional_encoding.parquet
│       ├── label_positional_encoding.parquet.origional
│       ├── pdf_net.h5
│       └── pdf_net.h5.origional
├── preferences
│   ├── __init__.py
│   └── pandas
│       └── __init__.py
├── preprocessing
│   ├── __init__.py
│   └── _images.py
├── reference_materials
│   ├── PDF32000_2008.pdf
│   └── pdf_reference_1-7.pdf
└── scripts
    ├── op_download.py
    └── streamlit
        ├── cardinal_data_capture.mp4
        └── ch_streamlit_app.py

15 directories, 56 files
```
