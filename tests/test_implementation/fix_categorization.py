"""
Fix for PhilosophicalOntology categorization issue
===============================================

The issue is in the _determine_primary_category method where domain_scores keys
might be strings or enum values, causing a type mismatch.
"""

def fix_categorization_issue():
    """
    Fixes the categorization issue in philosophical_ontology.py
    """
    
    # Read the file
    file_path = "/home/ty/Repositories/ai_workspace/openended-philosophy-mcp/openended_philosophy/semantic/philosophical_ontology.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the problematic method and fix it
    old_method = '''    def _determine_primary_category(self, semantic_analysis: SemanticAnalysis) -> PhilosophicalDomain:
        """Determine primary philosophical category through systematic analysis."""
        domain_scores = defaultdict(float)

        # Analyze concepts for domain indicators
        for concept in semantic_analysis.primary_concepts:
            if hasattr(concept, 'domain') and concept.domain:
                domain_scores[concept.domain] += concept.confidence_level

        # Analyze statement content for domain-specific terms
        for domain, indicators in self.domain_indicators.items():
            for _, terms in indicators.items():
                # Ideally, we would analyze the original statement text for domain-specific terms,
                # but since the raw statement is not available in SemanticAnalysis, we use concept terms instead.
                # To enable direct statement analysis, SemanticAnalysis would need to include the original text.
                concept_terms = [c.term.lower() for c in semantic_analysis.primary_concepts]
                matches = sum(1 for term in terms if any(term in ct for ct in concept_terms))
                domain_scores[domain] += matches * 0.1

        # Analyze semantic relations for domain preferences
        for relation in semantic_analysis.semantic_relations:
            if relation.relation_type in self.relation_types:
                mapping = self.relation_types[relation.relation_type]
                for domain in mapping.applicable_domains:
                    domain_scores[domain] += relation.confidence * 0.2

        # Find domain with highest score
        if domain_scores:
            primary_domain = max(domain_scores, key=lambda k: domain_scores[k])
            return primary_domain

        # Default fallback
        return PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE'''
    
    new_method = '''    def _determine_primary_category(self, semantic_analysis: SemanticAnalysis) -> PhilosophicalDomain:
        """Determine primary philosophical category through systematic analysis."""
        domain_scores = defaultdict(float)

        # Analyze concepts for domain indicators
        for concept in semantic_analysis.primary_concepts:
            if hasattr(concept, 'domain') and concept.domain:
                # Ensure we handle both enum values and strings
                if isinstance(concept.domain, PhilosophicalDomain):
                    domain_scores[concept.domain] += concept.confidence_level
                elif isinstance(concept.domain, str):
                    # Try to convert string to enum
                    try:
                        domain_enum = PhilosophicalDomain(concept.domain)
                        domain_scores[domain_enum] += concept.confidence_level
                    except ValueError:
                        # If string doesn't match enum, skip
                        continue

        # Analyze statement content for domain-specific terms
        for domain, indicators in self.domain_indicators.items():
            for _, terms in indicators.items():
                # Ideally, we would analyze the original statement text for domain-specific terms,
                # but since the raw statement is not available in SemanticAnalysis, we use concept terms instead.
                # To enable direct statement analysis, SemanticAnalysis would need to include the original text.
                concept_terms = [c.term.lower() for c in semantic_analysis.primary_concepts]
                matches = sum(1 for term in terms if any(term in ct for ct in concept_terms))
                domain_scores[domain] += matches * 0.1

        # Analyze semantic relations for domain preferences
        for relation in semantic_analysis.semantic_relations:
            if relation.relation_type in self.relation_types:
                mapping = self.relation_types[relation.relation_type]
                for domain in mapping.applicable_domains:
                    domain_scores[domain] += relation.confidence * 0.2

        # Find domain with highest score
        if domain_scores:
            primary_domain = max(domain_scores, key=lambda k: domain_scores[k])
            # Ensure we return a PhilosophicalDomain enum
            if isinstance(primary_domain, PhilosophicalDomain):
                return primary_domain
            elif isinstance(primary_domain, str):
                try:
                    return PhilosophicalDomain(primary_domain)
                except ValueError:
                    # If conversion fails, use default
                    return PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE
            else:
                # If it's neither string nor enum, use default
                return PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE

        # Default fallback
        return PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE'''
    
    # Replace the method
    if old_method in content:
        content = content.replace(old_method, new_method)
    else:
        print("Warning: Could not find exact method to replace. Method may have changed.")
        return False
    
    # Write back the fixed content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed categorization issue in philosophical_ontology.py")
    return True

if __name__ == "__main__":
    fix_categorization_issue()
