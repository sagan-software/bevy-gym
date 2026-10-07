//! Synchronous boundaries over Tokio file operations and child processes.

use std::io;
use std::path::{Path, PathBuf};
#[cfg(feature = "render")]
use std::process::ExitStatus;
use std::time::SystemTime;

#[cfg(feature = "render")]
use tokio::process::Command;
use tokio::runtime::{Builder, Runtime};

/// Build the current-thread runtime used by cold command-line operations.
fn runtime() -> io::Result<Runtime> {
    Builder::new_current_thread().enable_all().build()
}

/// Create a directory and all missing parents.
#[cfg(feature = "render")]
pub(crate) fn create_directory_all(path: &Path) -> io::Result<()> {
    let runtime = runtime()?;
    runtime.block_on(tokio::fs::create_dir_all(path))
}

/// Remove a directory and all of its contents.
#[cfg(any(test, feature = "render"))]
pub(crate) fn remove_directory_all(path: &Path) -> io::Result<()> {
    let runtime = runtime()?;
    runtime.block_on(tokio::fs::remove_dir_all(path))
}

/// Write a complete byte sequence to one file.
pub(crate) fn write(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    let runtime = runtime()?;
    runtime.block_on(tokio::fs::write(path, contents))
}

/// Read one complete file into an owned byte buffer.
#[cfg(test)]
pub(crate) fn read(path: &Path) -> io::Result<Vec<u8>> {
    let runtime = runtime()?;
    runtime.block_on(tokio::fs::read(path))
}

/// Return every direct child path of one directory.
pub(crate) fn directory_entries(path: &Path) -> io::Result<Vec<PathBuf>> {
    // Complete the blocking compatibility boundary before returning control to the caller.
    let runtime = runtime()?;
    runtime.block_on(async {
        let mut entries = tokio::fs::read_dir(path).await?;
        let mut paths = Vec::new();
        loop {
            let next_entry = entries.next_entry().await?;
            let Some(entry) = next_entry else {
                break;
            };
            paths.push(entry.path());
        }
        Ok(paths)
    })
}

/// Return the modification time for one path.
pub(crate) fn modified(path: &Path) -> io::Result<SystemTime> {
    let runtime = runtime()?;
    runtime.block_on(async { tokio::fs::metadata(path).await?.modified() })
}

/// Run one configured child process to completion.
#[cfg(feature = "render")]
pub(crate) fn command_status(command: &mut Command) -> io::Result<ExitStatus> {
    let runtime = runtime()?;
    runtime.block_on(command.status())
}
