"""Tests for ingestion entities."""

import pytest
import pandas as pd

from valtapy.ingestion.entities import DataSource, ReadResult, IngestionResult


class TestDataSource:
    """Test cases for DataSource entity."""
    
    def test_valid_data_source_creation(self):
        """Test creating a valid DataSource."""
        source = DataSource(
            path="/path/to/file.csv",
            format="csv",
            read_params={"encoding": "utf-8"}
        )
        
        assert source.path == "/path/to/file.csv"
        assert source.format == "csv"
        assert source.read_params == {"encoding": "utf-8"}
        
    def test_data_source_with_defaults(self):
        """Test DataSource with default read_params."""
        source = DataSource(path="file.csv", format="csv")
        
        assert source.read_params == {}
        
    def test_file_path_property(self):
        """Test file_path property returns Path object."""
        source = DataSource(path="test.csv", format="csv")
        
        file_path = source.file_path
        assert file_path.name == "test.csv"
        
    def test_empty_path_raises_error(self):
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="Path cannot be empty"):
            DataSource(path="", format="csv")
            
    def test_empty_format_raises_error(self):
        """Test that empty format raises ValueError."""
        with pytest.raises(ValueError, match="Format cannot be empty"):
            DataSource(path="file.csv", format="")


class TestReadResult:
    """Test cases for ReadResult entity."""
    
    def test_valid_read_result_creation(self):
        """Test creating a valid ReadResult."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        source = DataSource(path="test.csv", format="csv")
        metadata = {"shape": (2, 2)}
        
        result = ReadResult(
            data=df,
            source=source,
            metadata=metadata
        )
        
        assert result.data.equals(df)
        assert result.source == source
        assert result.metadata == metadata
        
    def test_read_result_with_default_metadata(self):
        """Test ReadResult with default empty metadata."""
        df = pd.DataFrame({"col": [1]})
        source = DataSource(path="test.csv", format="csv")
        
        result = ReadResult(data=df, source=source)
        
        assert result.metadata == {}
        
    def test_none_data_raises_error(self):
        """Test that None data raises ValueError."""
        source = DataSource(path="test.csv", format="csv")
        
        with pytest.raises(ValueError, match="Data cannot be None"):
            ReadResult(data=None, source=source)


class TestIngestionResult:
    """Test cases for IngestionResult entity."""
    
    def test_valid_ingestion_result_creation(self):
        """Test creating a valid IngestionResult."""
        df1 = pd.DataFrame({"col": [1, 2]})
        df2 = pd.DataFrame({"col": [3, 4]})
        
        source1 = DataSource(path="real.csv", format="csv")
        source2 = DataSource(path="synth.csv", format="csv")
        
        read1 = ReadResult(data=df1, source=source1)
        read2 = ReadResult(data=df2, source=source2)
        
        result = IngestionResult(
            real_data=read1,
            synth_data=read2,
            processing_time=1.5,
            issues=["Warning: different sizes"]
        )
        
        assert result.real_data == read1
        assert result.synth_data == read2
        assert result.processing_time == 1.5
        assert result.issues == ["Warning: different sizes"]
        
    def test_ingestion_result_with_defaults(self):
        """Test IngestionResult with default values."""
        df = pd.DataFrame({"col": [1]})
        source = DataSource(path="test.csv", format="csv")
        read_result = ReadResult(data=df, source=source)
        
        result = IngestionResult(
            real_data=read_result,
            synth_data=read_result
        )
        
        assert result.processing_time == 0.0
        assert result.issues == []