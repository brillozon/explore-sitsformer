#!/bin/bash

# GitHub Labels Setup Script for SITSFormer
# This script creates a comprehensive set of labels for issue and PR management
# Run this script with: bash .github/scripts/setup-labels.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed. Please install it first.${NC}"
    echo "Visit: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${RED}Error: Not authenticated with GitHub CLI. Please run 'gh auth login' first.${NC}"
    exit 1
fi

echo -e "${BLUE}🏷️  Setting up GitHub labels for SITSFormer repository...${NC}"

# Function to create or update a label
create_label() {
    local name="$1"
    local description="$2"
    local color="$3"
    
    # Try to create the label, if it exists, update it
    if gh label create "$name" --description "$description" --color "$color" 2>/dev/null; then
        echo -e "${GREEN}✅ Created label: $name${NC}"
    else
        # Label might exist, try to edit it
        if gh label edit "$name" --description "$description" --color "$color" 2>/dev/null; then
            echo -e "${YELLOW}📝 Updated label: $name${NC}"
        else
            echo -e "${RED}❌ Failed to create/update label: $name${NC}"
        fi
    fi
}

echo -e "${BLUE}Creating priority labels...${NC}"
create_label "priority: critical" "Critical priority - immediate attention required" "B60205"
create_label "priority: high" "High priority - should be addressed soon" "D93F0B"
create_label "priority: medium" "Medium priority - normal timeline" "FBCA04"
create_label "priority: low" "Low priority - when time permits" "0E8A16"

echo -e "${BLUE}Creating type labels...${NC}"
create_label "type: bug" "Something isn't working correctly" "D73A4A"
create_label "type: feature" "New feature or enhancement request" "A2EEEF"
create_label "type: documentation" "Improvements or additions to documentation" "0075CA"
create_label "type: performance" "Performance improvement or optimization" "FF6B6B"
create_label "type: refactor" "Code refactoring without functional changes" "FFEB3B"
create_label "type: maintenance" "Routine maintenance and housekeeping" "FEF2C0"
create_label "type: security" "Security-related issue or improvement" "B60205"
create_label "type: question" "General question or discussion" "D876E3"

echo -e "${BLUE}Creating status labels...${NC}"
create_label "status: triage" "Needs initial review and categorization" "FFFFFF"
create_label "status: investigating" "Issue is being investigated" "C5DEF5"
create_label "status: in progress" "Work is currently in progress" "F9D0C4"
create_label "status: blocked" "Blocked by external dependency or decision" "B60205"
create_label "status: needs review" "Ready for review" "0052CC"
create_label "status: needs testing" "Requires testing before completion" "C2E0C6"
create_label "status: duplicate" "This issue or PR already exists" "CFD3D7"
create_label "status: wontfix" "This will not be worked on" "FFFFFF"

echo -e "${BLUE}Creating component labels...${NC}"
create_label "component: model" "Related to model architecture and implementation" "FF6B35"
create_label "component: data" "Related to data loading, processing, or pipelines" "4ECDC4"
create_label "component: training" "Related to training infrastructure and optimization" "45B7D1"
create_label "component: evaluation" "Related to model evaluation and metrics" "96CEB4"
create_label "component: config" "Related to configuration management" "FFEAA7"
create_label "component: utils" "Related to utility functions and helpers" "DDA0DD"
create_label "component: api" "Related to public API and interfaces" "74B9FF"
create_label "component: cli" "Related to command-line interface" "A29BFE"

echo -e "${BLUE}Creating environment labels...${NC}"
create_label "env: development" "Development environment specific" "E1F5FE"
create_label "env: testing" "Testing environment specific" "FFF3E0"
create_label "env: production" "Production environment specific" "FFEBEE"
create_label "env: docker" "Docker-related issue" "2196F3"
create_label "env: gpu" "GPU-specific issue" "FF5722"
create_label "env: cpu" "CPU-specific issue" "607D8B"

echo -e "${BLUE}Creating workflow labels...${NC}"
create_label "workflow: needs design" "Requires design discussion before implementation" "FFC107"
create_label "workflow: good first issue" "Good for newcomers to the project" "7057FF"
create_label "workflow: help wanted" "Extra attention is needed from the community" "008672"
create_label "workflow: breaking change" "Introduces breaking changes" "B60205"
create_label "workflow: backwards compatible" "Maintains backward compatibility" "0E8A16"

echo -e "${BLUE}Creating dependency labels...${NC}"
create_label "dependencies" "Pull requests that update a dependency file" "0366D6"
create_label "python" "Python dependency updates" "3776AB"
create_label "github-actions" "GitHub Actions dependency updates" "000000"
create_label "docker" "Docker dependency updates" "2496ED"

echo -e "${BLUE}Creating CI/CD labels...${NC}"
create_label "ci/cd" "Related to CI/CD pipelines and automation" "28A745"
create_label "release" "Related to release management" "FF6347"
create_label "benchmark" "Triggers performance benchmarks" "FF6B6B"
create_label "memory-profile" "Triggers memory profiling" "9C27B0"

echo -e "${BLUE}Creating research labels...${NC}"
create_label "research" "Research-related feature or investigation" "E91E63"
create_label "experimental" "Experimental feature - use with caution" "FF9800"
create_label "paper" "Related to academic paper or publication" "673AB7"

echo -e "${BLUE}Creating size labels (for PRs)...${NC}"
create_label "size: XS" "Very small change (1-10 lines)" "C2E0C6"
create_label "size: S" "Small change (10-50 lines)" "BFDADC"
create_label "size: M" "Medium change (50-250 lines)" "FEF2C0"
create_label "size: L" "Large change (250-1000 lines)" "F9D71C"
create_label "size: XL" "Very large change (1000+ lines)" "E99695"

echo -e "${BLUE}Creating domain-specific labels...${NC}"
create_label "satellite imagery" "Related to satellite image processing" "4CAF50"
create_label "time series" "Related to time series analysis" "2196F3"
create_label "transformer" "Related to transformer architecture" "9C27B0"
create_label "attention" "Related to attention mechanisms" "E91E63"
create_label "remote sensing" "Related to remote sensing applications" "FF9800"

echo -e "${BLUE}Creating community labels...${NC}"
create_label "community" "Community-driven discussion or contribution" "FFC107"
create_label "hacktoberfest" "Eligible for Hacktoberfest contributions" "FF6347"
create_label "first-timers-only" "Reserved for first-time contributors" "FFEB3B"

echo -e "${GREEN}🎉 GitHub labels setup completed successfully!${NC}"
echo -e "${BLUE}📝 Labels have been created/updated in your repository.${NC}"
echo -e "${YELLOW}💡 You can view all labels at: https://github.com/$(gh repo view --json owner,name -q '.owner.login + "/" + .name')/labels${NC}"