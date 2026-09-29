#!/bin/bash

# Find the absolute path of the MoDrAg_SYS1 directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if modrag_cli.py exists in the code subdirectory
if [ ! -f "$SCRIPT_DIR/code/modrag_cli.py" ]; then
  echo "Error: modrag_cli.py not found in $SCRIPT_DIR/code/"
  exit 1
fi

CLI_SCRIPT="$SCRIPT_DIR/code/modrag_cli.py"

# Prefer the project virtualenv's interpreter when it exists (the tool
# dependencies — rdkit, dockstring, torch, ... — are installed there);
# fall back to the system python3 otherwise.
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

# Detect the user's current shell
CURRENT_SHELL="$(basename "$SHELL")"

# Function to add alias to a shell config file
add_alias_to_file() {
  local config_file=$1
  local shell_name=$2

  # Create file if it doesn't exist
  if [ ! -f "$config_file" ]; then
    touch "$config_file"
    echo "Created $config_file"
  fi

  # Remove any prior modrag definition — an old `alias modrag=`, the
  # `unalias modrag` guard, or a `modrag()` function (single-line form
  # written by this script).
  if grep -qE "^(alias modrag=|unalias modrag|modrag\(\))" "$config_file"; then
    echo "Existing modrag definition found in $config_file — replacing it"
    sed -i '' -e '/alias modrag=/d' -e '/^unalias modrag/d' -e '/^modrag()/d' "$config_file" \
      || sed -i -e '/alias modrag=/d' -e '/^unalias modrag/d' -e '/^modrag()/d' "$config_file"
  fi

  # Add the new definition as a shell FUNCTION that cds into code/ in a
  # subshell before running. This is required because modrag_cli.py and
  # the tool nodes use CWD-relative paths (../images, ../scratch,
  # ../vault, ../pdb_files, ../data) that only resolve correctly when
  # CWD is code/. The subshell ( ... ) means the cd is discarded when
  # the CLI exits, so the user's terminal stays put. The leading
  # `unalias modrag 2>/dev/null` clears any stale alias left in a live
  # shell from the old `alias modrag=` form; without it, re-sourcing
  # the config triggers "defining function based on alias `modrag'".
  echo "unalias modrag 2>/dev/null" >> "$config_file"
  echo "modrag() { ( cd '$SCRIPT_DIR/code' && '$PYTHON' modrag_cli.py \"\$@\" ); }" >> "$config_file"

  echo "Added modrag function to $config_file:"
  echo "  modrag() { ( cd '$SCRIPT_DIR/code' && '$PYTHON' modrag_cli.py \"\$@\" ); }"
}

# Apply to detected shell's config file
if [ "$CURRENT_SHELL" = "zsh" ]; then
  add_alias_to_file ~/.zshrc "zsh"
  source ~/.zshrc
  echo "Sourced ~/.zshrc"
elif [ "$CURRENT_SHELL" = "bash" ]; then
  add_alias_to_file ~/.bashrc "bash"
  source ~/.bashrc
  echo "Sourced ~/.bashrc"
else
  echo "Unknown shell: $CURRENT_SHELL"
  echo "Attempting to add to both .zshrc and .bashrc..."
  add_alias_to_file ~/.zshrc "zsh"
  add_alias_to_file ~/.bashrc "bash"
  echo "Note: Please manually source your shell config file"
fi

echo ""
echo "You can now use 'modrag' from anywhere to run MoDrAg System 1!"