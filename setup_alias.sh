#!/bin/bash

# Install a `modragsys1` shell function for MoDrAg System 1.
#
# The command name is deliberately NOT `modrag`: that name belongs to
# MoDrAg_CLI (MauricioCafiero/MoDrAg), whose own setup_alias.sh defines
# a `modrag()` function whose cleanup sed (~ /^modrag()/d,
# ^unalias modrag/d) removes any `modrag` definition it finds. If both
# installers used the name, whichever ran last would silently break the
# other one. With `modragsys1` here, both agents can be installed and
# coexist: `modrag` runs MoDrAg_CLI, `modragsys1` runs System 1. (That
# cleanup pattern is a prefix match, so re-running MoDrAg_CLI's
# installer can still drop our `unalias modragsys1` guard line —
# harmless, it only loses stale-alias cleanup, never the function.)

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

  # Remove any previous definition written by this script (identified
  # by this script's cd path) or inside its own marker block, so
  # re-running it updates the path instead of accumulating entries.
  # Foreign definitions (MoDrAg_CLI's `modrag`, hand-written aliases)
  # are never touched.
  local tmp
  tmp="${config_file}.modragtmp.$$"
  awk -v sd="$SCRIPT_DIR" '
    /^# >>> modragsys1 >>>$/ {skip=1; next}
    /^# <<< modragsys1 <<<$/ {skip=0; next}
    skip==1 {next}
    (/modragsys1\(\)/ && $0 ~ sd) {next}
    (/alias modragsys1=/ && $0 ~ sd) {next}
    (/unalias modragsys1/ && !guard) {guard=1; next}
    {print}
  ' "$config_file" > "$tmp" && cat "$tmp" > "$config_file" && rm -f "$tmp"

  # Add the new definition as a shell FUNCTION that cds into code/ in a
  # subshell before running. This is required because modrag_cli.py and
  # the tool nodes use CWD-relative paths (../images, ../scratch,
  # ../vault, ../pdb_files, ../data) that only resolve correctly when
  # CWD is code/. The subshell ( ... ) means the cd is discarded when
  # the CLI exits, so the user's terminal stays put. The leading
  # `unalias` clears any stale alias left in a live shell from an old
  # `alias modragsys1=` form; without it, re-sourcing the config
  # triggers "defining function based on alias `modragsys1'". The
  # `f` flag keeps just the first such guard line.
  {
    echo "# >>> modragsys1 >>>"
    echo "unalias modragsys1 2>/dev/null"
    echo "modragsys1() { ( cd '$SCRIPT_DIR/code' && '$PYTHON' modrag_cli.py \"\$@\" ); }"
    echo "# <<< modragsys1 <<<"
  } >> "$config_file"

  echo "Added modragsys1() function to $config_file:"
  echo "  modragsys1() { ( cd '$SCRIPT_DIR/code' && '$PYTHON' modrag_cli.py \"\$@\" ); }"
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
echo 'You can now use `modragsys1` from anywhere to run MoDrAg System 1 (`modrag`, if you also installed MoDrAg_CLI, runs MoDrAg_CLI).'