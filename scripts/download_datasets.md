# Dataset download & layout

Place each dataset as an **ImageFolder** tree:

```text
data/
  plant_disease_expert/
    ClassA/
      img1.jpg
      ...
    ClassB/
      ...
  plantvillage/
    ...
  tea_leaf/
    ...
```

## 1. Plant Disease Expert (paper dataset)

Kaggle: https://www.kaggle.com/datasets/sadmansakibmahi/plant-disease-expert

```bash
kaggle datasets download -d sadmansakibmahi/plant-disease-expert -p data/raw --unzip
```

Point `data/plant_disease_expert` at the folder that contains class subfolders (often a nested `Image Data` / `train` directory inside the zip). You can junction/symlink or copy:

```powershell
# Example — adjust the inner path after inspecting the unzip contents
New-Item -ItemType Junction -Path "data\plant_disease_expert" -Target "data\raw\<inner_folder_with_classes>"
```

## 2. PlantVillage

Common Kaggle mirrors (pick one that gives 38 class folders):

```bash
kaggle datasets download -d emmarex/plantdisease -p data/raw_pv --unzip
# or
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw_pv --unzip
```

Link the folder that contains the 38 class directories to `data/plantvillage`.

## 3. Tea Leaf Disease

Example Kaggle sets:

```bash
kaggle datasets download -d shashwatwork/identifying-disease-in-tea-leafs -p data/raw_tea --unzip
# or
kaggle datasets download -d saife245/tea-sickness-dataset -p data/raw_tea --unzip
```

Link class folders to `data/tea_leaf`.

## Kaggle credentials

1. Create an API token at https://www.kaggle.com/settings → **Create New Token**.
2. Place `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows).
3. `pip install kaggle`

## Smoke-test a folder

```bash
python -m src.data --root data/plant_disease_expert --image-size 224
```
