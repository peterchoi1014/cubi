use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub path: PathBuf,
}

pub fn load_skills() -> Vec<Skill> {
    let Some(dir) = skills_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Ok(body) = fs::read_to_string(&path) else {
            continue;
        };
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let description = body
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(|line| line.trim_start_matches("# ").to_string())
            .unwrap_or_else(|| name.clone());
        skills.push(Skill {
            name,
            description,
            body,
            path,
        });
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

pub fn skills_dir() -> Option<PathBuf> {
    crate::sessions::cubi_dir().map(|d| d.join("skills"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skills_dir_uses_cubi_home() {
        crate::compat::test_env::with_cubi_home(|cubi_home, other_home| {
            let path = skills_dir().expect("skills dir");
            assert_eq!(path, cubi_home.join(".cubi").join("skills"));
            assert!(!path.starts_with(other_home));
        });
    }
}
