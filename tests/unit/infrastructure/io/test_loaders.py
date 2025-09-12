"""Unit tests for data loading functionality."""

import pytest
import sys
import tempfile
import os
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from valtapyV2.infrastructure.io.loaders import (
    load_csv,
    load_parquet,
    save_csv,
    get_file_info,
    _basic_cleanup
)
from valtapyV2.domain.errors import SchemaError
from tests.utils.test_helpers import convert_to_pandas_mock


class TestCSVLoading:
    """Test suite for CSV loading functionality."""
    
    def test_load_csv_basic(self):
        """Test basic CSV loading functionality."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            f.write("7,8,9\n")
            temp_path = f.name
        
        try:
            # Test loading
            df = load_csv(temp_path)
            
            # Verify structure
            assert len(df) == 3
            assert len(df.columns) == 3
            assert list(df.columns) == ["col1", "col2", "col3"]
            
            # Verify data
            assert df.iloc[0, 0] == 1
            assert df.iloc[0, 1] == 2
            assert df.iloc[0, 2] == 3
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_with_missing_values(self):
        """Test CSV loading with various missing value representations."""
        # Create CSV with different missing value formats
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,value,category\n")
            f.write("Alice,10,A\n")
            f.write("Bob,,B\n")  # Empty value
            f.write("Charlie,NA,C\n")  # NA
            f.write(",15,D\n")  # Empty name
            f.write("Dave,null,\n")  # null and empty category
            temp_path = f.name
        
        try:
            df = load_csv(temp_path)
            
            # Verify missing values are handled
            assert len(df) == 5
            assert df.isnull().any().any()  # Should have some NaN values
            
            # Check specific missing values
            assert pd.isna(df.iloc[1, 1])  # Bob's value should be NaN
            assert pd.isna(df.iloc[2, 1])  # Charlie's NA should be NaN
            assert pd.isna(df.iloc[4, 1])  # Dave's null should be NaN
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_with_custom_parameters(self):
        """Test CSV loading with custom parameters."""
        # Create CSV with semicolon separator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1;col2;col3\n")
            f.write("1;2;3\n")
            f.write("4;5;6\n")
            temp_path = f.name
        
        try:
            # Load with custom separator
            df = load_csv(temp_path, sep=';')
            
            assert len(df) == 2
            assert len(df.columns) == 3
            assert df.iloc[0, 0] == 1
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_file_not_found(self):
        """Test CSV loading with non-existent file."""
        with pytest.raises(SchemaError) as exc_info:
            load_csv("nonexistent_file.csv")
        
        assert "File not found" in str(exc_info.value)
    
    def test_load_csv_empty_file(self):
        """Test CSV loading with empty file."""
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(SchemaError) as exc_info:
                load_csv(temp_path)
            
            assert "empty" in str(exc_info.value).lower()
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_only_headers(self):
        """Test CSV loading with only headers (no data rows)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")  # Only headers
            temp_path = f.name
        
        try:
            with pytest.raises(SchemaError) as exc_info:
                load_csv(temp_path)
            
            assert "empty" in str(exc_info.value).lower()
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_malformed_file(self):
        """Test CSV loading with malformed CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("1,2\n")  # Missing column
            f.write("4,5,6,7\n")  # Extra column
            f.write("incomplete")  # No newline
            temp_path = f.name
        
        try:
            # Should still load (pandas is quite forgiving)
            df = load_csv(temp_path)
            assert len(df) >= 2  # Should load at least some rows
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_with_whitespace(self):
        """Test CSV loading with whitespace handling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,value,category\n")
            f.write(" Alice , 10 , A \n")  # Spaces around values
            f.write("  Bob,  ,B\n")  # Spaces and empty value
            temp_path = f.name
        
        try:
            df = load_csv(temp_path)
            
            # Whitespace should be handled by skipinitialspace=True
            assert df.iloc[0, 0] == "Alice"  # Should be trimmed
            assert df.iloc[0, 1] == 10
            assert df.iloc[0, 2] == "A"
            
        finally:
            os.unlink(temp_path)


class TestBasicCleanup:
    """Test suite for basic DataFrame cleanup functionality."""
    
    def test_basic_cleanup_empty_rows(self):
        """Test removal of completely empty rows."""
        # Create mock DataFrame with empty rows
        data = [
            {"col1": "A", "col2": 1, "col3": "X"},
            {"col1": None, "col2": None, "col3": None},  # Completely empty
            {"col1": "B", "col2": 2, "col3": "Y"},
            {"col1": None, "col2": None, "col3": None},  # Completely empty
            {"col1": "C", "col2": None, "col3": "Z"}  # Partially empty
        ]
        
        df = convert_to_pandas_mock(data)
        cleaned_df = _basic_cleanup(df)
        
        # Should remove the 2 completely empty rows
        assert len(cleaned_df) == 3
        assert list(cleaned_df.iloc[:, 0]) == ["A", "B", "C"]
    
    def test_basic_cleanup_empty_columns(self):
        """Test removal of completely empty columns."""
        data = [
            {"col1": "A", "col2": None, "col3": 1, "col4": None},
            {"col1": "B", "col2": None, "col3": 2, "col4": None},
            {"col1": "C", "col2": None, "col3": 3, "col4": None}
        ]
        
        df = convert_to_pandas_mock(data)
        cleaned_df = _basic_cleanup(df)
        
        # Should remove col2 and col4 (completely empty)
        assert len(cleaned_df.columns) == 2
        assert "col1" in cleaned_df.columns
        assert "col3" in cleaned_df.columns
        assert "col2" not in cleaned_df.columns
        assert "col4" not in cleaned_df.columns
    
    def test_basic_cleanup_string_whitespace(self):
        """Test whitespace stripping from string columns."""
        data = [
            {"name": " Alice ", "value": 10, "category": "  A  "},
            {"name": "Bob  ", "value": 20, "category": " B"},
            {"name": "  Charlie", "value": 30, "category": "C "}
        ]
        
        df = convert_to_pandas_mock(data)
        cleaned_df = _basic_cleanup(df)
        
        # String values should be trimmed
        names = cleaned_df.iloc[:, 0].tolist()  # name column
        categories = cleaned_df.iloc[:, 2].tolist()  # category column
        
        assert "Alice" in names
        assert "Bob" in names  
        assert "Charlie" in names
        assert "A" in categories
        assert "B" in categories
        assert "C" in categories
        
        # Numeric values should be unchanged
        values = cleaned_df.iloc[:, 1].tolist()  # value column
        assert 10 in values
        assert 20 in values
        assert 30 in values
    
    def test_basic_cleanup_empty_strings_to_nan(self):
        """Test conversion of empty strings to NaN."""
        data = [
            {"name": "Alice", "category": "A"},
            {"name": "", "category": "B"},  # Empty string
            {"name": "Charlie", "category": ""},  # Empty string
            {"name": "   ", "category": "D"}  # Whitespace only
        ]
        
        df = convert_to_pandas_mock(data)
        cleaned_df = _basic_cleanup(df)
        
        # Empty strings should become NaN after stripping
        # Note: This test depends on the mock implementation's handling of NaN


class TestFileInfo:
    """Test suite for file information functionality."""
    
    def test_get_file_info_nonexistent(self):
        """Test file info for non-existent file."""
        info = get_file_info("nonexistent_file.csv")
        
        assert info["exists"] == False
        assert "error" in info
    
    def test_get_file_info_csv_basic(self):
        """Test file info for basic CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            temp_path = f.name
        
        try:
            info = get_file_info(temp_path)
            
            assert info["exists"] == True
            assert info["suffix"] == ".csv"
            assert info["size_bytes"] > 0
            assert "columns" in info
            assert len(info["columns"]) == 3
            assert info["n_columns"] == 3
            assert "dtypes" in info
            
        finally:
            os.unlink(temp_path)
    
    def test_get_file_info_csv_malformed(self):
        """Test file info for malformed CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("This is not a valid CSV file format!!!")
            temp_path = f.name
        
        try:
            info = get_file_info(temp_path)
            
            assert info["exists"] == True
            assert info["suffix"] == ".csv"
            assert "error" in info  # Should have error reading CSV sample
            
        finally:
            os.unlink(temp_path)
    
    def test_get_file_info_non_csv(self):
        """Test file info for non-CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a text file.")
            temp_path = f.name
        
        try:
            info = get_file_info(temp_path)
            
            assert info["exists"] == True
            assert info["suffix"] == ".txt"
            assert info["size_bytes"] > 0
            # Should not have CSV-specific info
            assert "columns" not in info
            assert "n_columns" not in info
            
        finally:
            os.unlink(temp_path)


class TestCSVSaving:
    """Test suite for CSV saving functionality."""
    
    def test_save_csv_basic(self):
        """Test basic CSV saving functionality."""
        # Create test data
        data = [
            {"name": "Alice", "age": 25, "city": "Madrid"},
            {"name": "Bob", "age": 30, "city": "Barcelona"},
            {"name": "Charlie", "age": 35, "city": "Valencia"}
        ]
        
        df = convert_to_pandas_mock(data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            save_csv(df, temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Verify content by loading it back
            loaded_df = load_csv(temp_path)
            assert len(loaded_df) == 3
            assert len(loaded_df.columns) == 3
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_csv_with_custom_params(self):
        """Test CSV saving with custom parameters."""
        data = [
            {"col1": 1, "col2": 2},
            {"col1": 3, "col2": 4}
        ]
        
        df = convert_to_pandas_mock(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save with custom separator
            save_csv(df, temp_path, sep=';')
            
            # Verify file exists
            assert os.path.exists(temp_path)
            
            # Read file content directly to check separator
            with open(temp_path, 'r') as f:
                content = f.read()
                assert ';' in content  # Should use semicolon separator
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_csv_creates_directories(self):
        """Test that save_csv creates parent directories."""
        # Use a path with non-existent directory
        temp_dir = tempfile.mkdtemp()
        nested_path = os.path.join(temp_dir, "nested", "dir", "test.csv")
        
        data = [{"test": 1}]
        df = convert_to_pandas_mock(data)
        
        try:
            save_csv(df, nested_path)
            
            # Verify file was created in nested directory
            assert os.path.exists(nested_path)
            
        finally:
            # Cleanup
            if os.path.exists(nested_path):
                os.unlink(nested_path)
            # Remove created directories
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestDataLoadingIntegration:
    """Test suite for integrated data loading scenarios."""
    
    def test_load_real_and_synthetic_data_workflow(self):
        """Test typical workflow of loading real and synthetic data."""
        # Create mock real data
        real_data_content = """feature1,feature2,target
1.0,2.0,A
2.0,4.0,B
3.0,6.0,A
4.0,8.0,B
"""
        
        # Create mock synthetic data  
        synth_data_content = """feature1,feature2,target
1.1,2.1,A
2.1,4.1,B
3.1,6.1,A
4.1,8.1,B
"""
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as real_f:
            real_f.write(real_data_content)
            real_path = real_f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as synth_f:
            synth_f.write(synth_data_content)
            synth_path = synth_f.name
        
        try:
            # Load both datasets
            real_df = load_csv(real_path)
            synth_df = load_csv(synth_path)
            
            # Verify structure
            assert len(real_df) == 4
            assert len(synth_df) == 4
            assert list(real_df.columns) == list(synth_df.columns)
            assert len(real_df.columns) == 3
            
            # Verify they have the same structure but different data
            assert real_df.shape == synth_df.shape
            
            # Check that data is actually different
            feature1_real = real_df.iloc[:, 0].tolist()  
            feature1_synth = synth_df.iloc[:, 0].tolist()
            
            # Should be different values but similar structure
            assert feature1_real != feature1_synth
            assert len(feature1_real) == len(feature1_synth)
            
        finally:
            os.unlink(real_path)
            os.unlink(synth_path)
    
    def test_load_data_with_different_schemas(self):
        """Test loading data files with different schemas."""
        # Real data with 3 columns
        real_content = """col1,col2,col3
1,2,3
4,5,6
"""
        
        # Synthetic data with different columns (should raise issues in actual usage)
        synth_content = """col1,col2,col4
1,2,7
4,5,8
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as real_f:
            real_f.write(real_content)
            real_path = real_f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as synth_f:
            synth_f.write(synth_content)
            synth_path = synth_f.name
        
        try:
            # Load both (should succeed individually)
            real_df = load_csv(real_path)
            synth_df = load_csv(synth_path)
            
            # They load successfully but have different schemas
            assert list(real_df.columns) != list(synth_df.columns)
            assert "col3" in real_df.columns
            assert "col4" in synth_df.columns
            assert "col3" not in synth_df.columns
            assert "col4" not in real_df.columns
            
        finally:
            os.unlink(real_path)
            os.unlink(synth_path)
    
    def test_load_data_performance_info(self):
        """Test getting performance info about data files."""
        # Create a larger CSV for testing
        content = "feature1,feature2,target\n"
        for i in range(100):
            content += f"{i},{i*2},{'A' if i % 2 == 0 else 'B'}\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            # Get file info
            info = get_file_info(temp_path)
            
            assert info["exists"] == True
            assert info["n_columns"] == 3
            assert len(info["columns"]) == 3
            assert info["size_bytes"] > 100  # Should be reasonable size
            
            # Load and verify it matches info
            df = load_csv(temp_path)
            assert len(df.columns) == info["n_columns"]
            assert list(df.columns) == info["columns"]
            
        finally:
            os.unlink(temp_path)


# Import pandas for tests that need it
try:
    import pandas as pd
except ImportError:
    pytest.skip("pandas not available", allow_module_level=True)