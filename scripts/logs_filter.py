import re


def parse_fine_tuning_logs(log_text):
    """
    Parse fine-tuning logs and extract key metrics.

    Returns a list of dictionaries containing:
    - proportion: the fine-tuning proportion
    - samples: number of samples selected
    - loss_after: Average Loss on Test Set
    - fid_after: FID on Test Set
    - loss_0_after: Average Loss on Test (only 0) Set
    - fid_0_after: FID on Test (only 0) Set
    """

    results = []

    # Split the log into sections for each fine-tuning run
    sections = re.split(r"Fine-tuning with random .+? proportion\.\.\.", log_text)

    # Get all the proportion matches
    proportion_matches = re.findall(
        r"Fine-tuning with random (.+?) proportion\.\.\.", log_text
    )

    for i, section in enumerate(sections[1:]):  # Skip first empty section
        if i < len(proportion_matches):
            proportion = float(proportion_matches[i])

            # Extract number of samples
            samples_match = re.search(r"Selected (\d+) random samples", section)
            samples = int(samples_match.group(1)) if samples_match else None

            # Extract metrics
            loss_match = re.search(r"Avg Loss on Test Set: ([\d.]+)", section)
            fid_match = re.search(r"FID on Test Set: ([\d.]+)", section)
            loss_0_match = re.search(
                r"Avg Loss on Test \(only 0\) Set: ([\d.]+)", section
            )
            fid_0_match = re.search(r"FID on Test \(only 0\) Set: ([\d.]+)", section)

            result = {
                "proportion": proportion,
                "samples": samples,
                "loss_after": float(loss_match.group(1)) if loss_match else None,
                "fid_after": float(fid_match.group(1)) if fid_match else None,
                "loss_0_after": float(loss_0_match.group(1)) if loss_0_match else None,
                "fid_0_after": float(fid_0_match.group(1)) if fid_0_match else None,
            }

            results.append(result)

    return results


def print_results_as_lists(results):
    """Print results as Python lists"""

    # Extract each metric into separate lists
    proportions = [r["proportion"] for r in results]
    samples = [r["samples"] for r in results]
    loss_after = [r["loss_after"] for r in results]
    fid_after = [r["fid_after"] for r in results]
    loss_0_after = [r["loss_0_after"] for r in results]
    fid_0_after = [r["fid_0_after"] for r in results]

    print("# Proportions")
    print(f"proportions = {proportions}")
    print()

    print("# Number of samples")
    print(f"samples = {samples}")
    print()

    print("# Loss After")
    print(f"loss_after = {loss_after}")
    print()

    print("# FID After")
    print(f"fid_after = {fid_after}")
    print()

    print("# Loss 0 After")
    print(f"loss_0_after = {loss_0_after}")
    print()

    print("# FID 0 After")
    print(f"fid_0_after = {fid_0_after}")


# Example usage:
if __name__ == "__main__":
    # Read the log file
    with open("results/paste.txt", "r") as file:
        log_content = file.read()

    # Parse the logs
    parsed_results = parse_fine_tuning_logs(log_content)

    # Print results as Python lists
    print_results_as_lists(parsed_results)
