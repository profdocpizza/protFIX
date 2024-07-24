import yaml
import os
import ampal
from ampal import Polypeptide
import pandas as pd
import numpy as np

# show more rows in pandas
pd.set_option("display.max_rows", 200)
import os
import copy
import pickle

import yaml
from isambard.modelling.scwrl import pack_side_chains_scwrl


def load_tasks(path_to_tasks_yaml):
    with open(path_to_tasks_yaml, "r") as file:
        tasks = yaml.safe_load(file)["tasks"]
    return tasks


def get_root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def find_best_alignment_extend(df, full_sequence):
    seq_length = len(full_sequence)
    max_res_id = df.index.max()

    best_alignment = None
    best_start_pos = None
    best_match_count = 0
    total_non_na = df["mol_letter"].notna().sum()

    # Allow extension from -seq_length to max_res_id + seq_length
    for start_pos in range(-seq_length, max_res_id + 1):
        extended_index = range(
            min(start_pos, 1), max(start_pos + seq_length, max_res_id + 1)
        )
        temp_df = pd.DataFrame(index=extended_index).join(df)
        temp_df["complete_mol_letter"] = np.nan

        match_count = 0
        for i, letter in enumerate(full_sequence):
            pos = start_pos + i
            if pos in temp_df.index:
                if (
                    pd.notna(temp_df.at[pos, "mol_letter"])
                    and temp_df.at[pos, "mol_letter"] != letter
                ):
                    break
                temp_df.at[pos, "complete_mol_letter"] = letter
                if pd.notna(temp_df.at[pos, "mol_letter"]):
                    match_count += 1
        else:
            if match_count > best_match_count:
                best_alignment = temp_df
                best_start_pos = start_pos
                best_match_count = match_count

    if best_alignment is None or best_match_count == 0:
        raise ValueError(
            "No ideal match found between the provided sequence and the fragmented sequence with known gaps."
        )
    elif best_match_count == total_non_na:
        print("All fragmented residues matched perfectly to input sequence.")
    else:
        print(
            f"Best alignment found at start position: {best_start_pos}, matching {best_match_count} out of {total_non_na} known positions."
        )

    return best_alignment, best_start_pos


def clip_residue_ids(residue_df, first, last):
    return residue_df.loc[first:last]


def get_contigmap_contigs(residue_df: pd.DataFrame, chain_id: str):
    assert (
        "complete_mol_letter" in residue_df.columns
    ), "residue_df must contain a 'complete_mol_letter' column."
    # find first index at which complete_mol_letter is not na
    start_index = residue_df["complete_mol_letter"].first_valid_index()
    end_index = residue_df["complete_mol_letter"].last_valid_index()
    contigs_string = f"{chain_id}{start_index}-{end_index}"
    return contigs_string  # f'contigmap.contigs=[{contigs_string}]'


def chunk_sequential_numbers(numbers):
    if not numbers:
        return []
    chunks = []
    current_chunk = [numbers[0]]
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_chunk.append(numbers[i])
        else:
            chunks.append(current_chunk)
            current_chunk = [numbers[i]]
    chunks.append(current_chunk)
    return chunks


def get_contigmap_inpaint_str(residue_df: pd.DataFrame, chain_id: str):
    assert (
        "complete_mol_letter" in residue_df.columns
        and "mol_letter" in residue_df.columns
    ), "residue_df must contain a 'complete_mol_letter' and 'mol_letter' column."
    # find all row index numbers where mol_letter is na and complete_mol_letter is not na
    inpaint_indices = residue_df[
        residue_df["mol_letter"].isna() & residue_df["complete_mol_letter"].notna()
    ].index
    inpaint_chunks = chunk_sequential_numbers(list(inpaint_indices))
    inpaint_str = f""
    for inpaint_chunk in inpaint_chunks:
        chunk_inpaint_str = f"/{chain_id}{inpaint_chunk[0]}-{inpaint_chunk[-1]}"
        inpaint_str += chunk_inpaint_str
    inpaint_str = inpaint_str.strip("/")
    return inpaint_str  # f'contigmap.inpaint_str=[{inpaint_str}]'


def generate_rfdiffusion_command(
    residue_df,
    chain_id,
    input_path,
    output_prefix,
    rf_diffusion_install_dir,
    num_designs=1,
    deterministic=True,
):
    contigmap_contigs_string = get_contigmap_contigs(residue_df, chain_id)
    contigmap_inpaint_str_string = get_contigmap_inpaint_str(residue_df, chain_id)

    script = (
        f"python '{os.path.join(rf_diffusion_install_dir,'scripts', 'run_inference.py')}' "
        f"inference.output_prefix='{output_prefix}' "
        f"inference.input_pdb='{input_path}' "
        f"'contigmap.contigs=[{contigmap_contigs_string}]' "
        f"'contigmap.inpaint_str=[{contigmap_inpaint_str_string}]' "
        f"inference.num_designs={int(num_designs)} "
        + ("inference.deterministic=True " if deterministic else "")
    )

    return script


def create_and_save_residue_presets(monomers):
    residue_dict = {}
    keys = set([x.mol_letter for x in monomers])
    for key in keys:
        residue_dict[key] = copy.deepcopy(
            [x for x in monomers if x.mol_letter == key][0]
        )

    residue_dict
    # save residue_dict as pickle
    import pickle

    with open("/home/tadas/code/dreamer/data/ampal_residue_presets", "wb") as handle:
        pickle.dump(residue_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_residue_presets():
    with open("/home/tadas/code/dreamer/data/ampal_residue_presets", "rb") as handle:
        residue_dict = pickle.load(handle)
    return residue_dict


def save_commplete_chain_pdb(residue_df, monomers_list, chain_id, outfile_path):
    residue_presets = load_residue_presets()
    complete_residues = []
    for index, row in residue_df.iterrows():
        if pd.isna(row["mol_letter"]):
            # residue = Residue(atoms = list(residue_dict[row['complete_mol_letter']].get_atoms()),mol_code=row['complete_mol_letter'], monomer_id=index, )
            residue = copy.deepcopy(residue_presets[row["complete_mol_letter"]])
            residue.id = int(index)

        else:
            # find real monomer with the index
            # print(row)
            residue = [mon for mon in monomers_list if int(mon.id) == index][0]
        complete_residues.append(residue)

    complete_polypeptide = Polypeptide(complete_residues, polymer_id=chain_id)
    # create dir if not exists
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)

    # save pdb
    with open(outfile_path, "w") as f:
        f.write(complete_polypeptide.pdb)


def run_shell_script_in_conda_env(conda_env_name, script_path, run_directory="."):
    # Form the command to run, ensuring all in one subshell
    command = (
        "bash -c '"
        f"cd {run_directory} && "
        "source ~/anaconda3/etc/profile.d/conda.sh && "
        f"conda activate {conda_env_name} && "
        f"bash {script_path}'"
    )
    os.system(command)


def check_eligibility(assembly):
    polypeptides = [i for i in assembly if isinstance(i, Polypeptide)]
    assert (
        len(polypeptides) == 1
    ), "Expected only one polypeptide in the assembly. This feature only works for single-chain assemblies."


def generate_residue_df(monomers):
    mol_letters = [residue.mol_letter for residue in monomers]
    res_ids = [int(residue.id) for residue in monomers]
    # make a dataframe with sequence and res_ids where res_ids will be the index
    residue_df = pd.DataFrame({"mol_letter": mol_letters}, index=res_ids)
    max_res_id = max(res_ids)
    # Create a full index from 1 to max_res_id.
    full_index = range(1, max_res_id + 1)
    residue_df = residue_df.reindex(full_index)
    return residue_df


def align_input_sequence_to_residue_df(residue_df, sequence):
    residue_df, _ = find_best_alignment_extend(residue_df, sequence)
    residue_df = clip_residue_ids(residue_df, 1, 99999999)
    return residue_df


def run_structure_inpainting(task, residue_df, monomers, chain_id):

    mock_residues_pdb_path = os.path.join(
        task["output_dir"], f"{task['name']}_mock_residues.pdb"
    )
    save_commplete_chain_pdb(residue_df, monomers, chain_id, mock_residues_pdb_path)

    rf_diffusion_install_dir = "/home/tadas/code/RFdiffusion"
    output_prefix = f"{task['output_dir']}/{task['name']}_fixed_backbone"

    rfdiffusion_command = generate_rfdiffusion_command(
        residue_df,
        chain_id,
        mock_residues_pdb_path,
        output_prefix,
        rf_diffusion_install_dir,
        num_designs=1,
        deterministic=True,
    )

    # save rfdiffusion command as text file
    rfdiffusion_command_path = (
        f"{task['output_dir']}/{task['name']}_rfdiffusion_command.sh"
    )
    with open(rfdiffusion_command_path, "w") as f:
        f.write(rfdiffusion_command)

    run_shell_script_in_conda_env(
        "SE3nv", rfdiffusion_command_path, run_directory=task["output_dir"]
    )

    expected_output_file = output_prefix + "_0.pdb"
    return expected_output_file


def pack_side_chains_on_diffusion_output(diffusion_output_file_path, selected_sequence):

    fixed_backbone_assembly = ampal.load_pdb(diffusion_output_file_path)

    sequence_choice = "".join(selected_sequence)
    packed_fixed_sctructure = pack_side_chains_scwrl(
        fixed_backbone_assembly, [sequence_choice]
    )

    packed_file_path = diffusion_output_file_path.replace("_backbone_0", "")
    with open(packed_file_path, "w") as f:
        f.write(packed_fixed_sctructure.pdb)
    return packed_file_path
