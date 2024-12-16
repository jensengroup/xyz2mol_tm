import pandas as pd
from util import set_chiral_tag_on_smiles


def load_data(
    df_targets_path,
    df_smiles_path,
    test_ids,
    outlier_path,
    smiles_column_string,
    chirality=False,
):
    """Function for loading smiles into dataframes.

    Args:
        df_targets_path (str): Path to the file with the tmQMg calculated properties
        df_smiles_path (str): Path to the file with the smiles sets
        test_ids (str): Path to the file with the test set IDs
        outlier_path (str): Path to the files contining the outlier IDs
        smiles_column_string (str): The dataframe column label of the smiles set to extract
        chirality (bool): If chiral tags are added to the smiles

    Returns:
       df_training : df with the trainig set smiles
       df_test : df with the test set smiles
    """
    data_df = pd.read_csv(df_targets_path)
    smiles_df = pd.read_csv(df_smiles_path)
    print("Column string", smiles_column_string)

    # Only smiles where I have one
    smiles_df = smiles_df[
        ~(smiles_df[smiles_column_string].isnull())
        & (smiles_df[smiles_column_string] != "fail")
    ]

    print("Columns present in the data: ", data_df.columns, smiles_df.columns)

    df = pd.merge(smiles_df, data_df, how="left", left_on="IDs", right_on="id")

    # Exclude outliers
    with open(outlier_path, "r") as fh:
        outliers = fh.read().splitlines()
    df = df[~df.IDs.isin(outliers)]
    print("Df after exluding outlier:", len(df))

    # Encode the chirality into the SMILES.
    if chirality:
        chiral_tag_df = pd.read_csv("../SMILES_csvs/tmqmg_chiraltags.csv")

        df = pd.merge(df, chiral_tag_df, how="left", left_on="IDs", right_on="IDs")
        print("Removing rows that do not have chiral tag")
        print(f"Current length : {len(df)}")
        df = df[~df["chiralTag"].isnull()]
        print(f"After removal : {len(df)}")
        df[smiles_column_string] = df.apply(
            set_chiral_tag_on_smiles, args=(smiles_column_string,), axis=1
        )

    df_sub = df[
        [
            "IDs",
            smiles_column_string,
            "polarisability",
            "tzvp_homo_lumo_gap",
            "tzvp_dipole_moment",
        ]
    ]

    print(f"Length smiles df: {len(df_sub)}")

    # Select explicitly the same test set
    df_training = df_sub[~df_sub["IDs"].isin(test_ids)]
    df_test = df_sub[df_sub["IDs"].isin(test_ids)]

    print("Length test data: ", len(df_test), df_test[:3])
    print("Length training data: ", len(df_training), df_training[0:3])

    return df_training, df_test
